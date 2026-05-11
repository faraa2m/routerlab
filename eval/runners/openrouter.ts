// eval/runners/openrouter.ts — OpenRouter runner.
//
// Provider: OpenRouter (https://openrouter.ai/).
// API:      OpenAI-compatible REST at `/api/v1/chat/completions`.
// Auth:     bearer token from `OPENROUTER_API_KEY` env var.
//
// Pricing notes:
//   OpenRouter exposes a unified marketplace of models from many
//   providers. Models with a `:free` suffix are routed to a free
//   variant (typically rate-limited; see
//   https://openrouter.ai/docs/limits#free-models). For these we report
//   `usdCost: 0` per the OpenRouter pricing page. The catalog of free
//   models drifts — at any given submit time, verify against
//   https://openrouter.ai/models?supported_parameters=free and update
//   the `PRICING` table below.
//
//   Paid models can be added by appending their per-MTok rate from
//   https://openrouter.ai/models. We keep the default list free-only so
//   that running the eval harness against OpenRouter does not silently
//   spend money.
//
// Retry:
//   Shared `withRetries` helper from `_retry.ts` — up to 3 attempts with
//   jittered exponential backoff (1s, 2s, 4s ± 20%). 429 / 5xx / timeout
//   is retried; 4xx (other) is terminal.
//
// Security:
//   `OPENROUTER_API_KEY` never leaves the runner; provider headers are
//   not echoed. OpenRouter also recommends setting an optional
//   `HTTP-Referer` / `X-Title` for attribution — we set a static
//   `routerlab` so usage is identifiable but never leak the API key.

import { withRetries } from "./_retry.ts";
import {
  makeRunnerError,
  RUNNER_DEFAULTS,
  type Runner,
  type RunnerError,
  type RunRequest,
  type RunResponse,
} from "./_types.ts";

const PROVIDER_ID = "openrouter";
const ENDPOINT = "https://openrouter.ai/api/v1/chat/completions";
const ATTRIBUTION_TITLE = "routerlab";
const ATTRIBUTION_REFERER = "https://github.com/faraa2m/routerlab";

export interface OpenRouterPricingRow {
  inputUsdPerMtok: number;
  outputUsdPerMtok: number;
}

/**
 * Supported model pool. Models with `:free` are explicitly the free
 * variants OpenRouter exposes (rate-limited by the platform). Add paid
 * models here with their per-MTok pricing from
 * https://openrouter.ai/models when needed.
 *
 * Verify the free list at submit time against
 * https://openrouter.ai/models?supported_parameters=free — OpenRouter
 * rotates which models live on the free tier.
 */
const PRICING: Readonly<Record<string, OpenRouterPricingRow>> = {
  "meta-llama/llama-3.3-70b-instruct:free": {
    inputUsdPerMtok: 0,
    outputUsdPerMtok: 0,
  },
  "qwen/qwen-2.5-72b-instruct:free": {
    inputUsdPerMtok: 0,
    outputUsdPerMtok: 0,
  },
  "mistralai/mistral-small-3.1-24b-instruct:free": {
    inputUsdPerMtok: 0,
    outputUsdPerMtok: 0,
  },
};

interface OpenRouterChatResponse {
  choices?: Array<{
    message?: { content?: string | null };
    finish_reason?: string;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
  };
  error?: { message?: string; code?: number };
}

/**
 * Map an HTTP status code / abort to a typed `RunnerError`. Mirrors the
 * Anthropic / Groq / Together policy so the build orchestrator sees uniform
 * error shapes regardless of provider.
 */
function classifyOpenRouterError(
  status: number,
  body: string,
  isAbort: boolean,
): RunnerError {
  if (isAbort) {
    return makeRunnerError(
      PROVIDER_ID,
      "timeout",
      "OpenRouter request timed out",
      true,
    );
  }
  if (status === 401 || status === 403) {
    return makeRunnerError(
      PROVIDER_ID,
      "auth",
      `OpenRouter auth failed (${status})`,
      false,
    );
  }
  if (status === 429) {
    return makeRunnerError(
      PROVIDER_ID,
      "rate_limit",
      `OpenRouter rate limit (429): ${body.slice(0, 200)}`,
      true,
    );
  }
  if (status >= 500) {
    return makeRunnerError(
      PROVIDER_ID,
      "server",
      `OpenRouter server error (${status}): ${body.slice(0, 200)}`,
      true,
    );
  }
  if (status >= 400) {
    return makeRunnerError(
      PROVIDER_ID,
      "bad_request",
      `OpenRouter bad request (${status}): ${body.slice(0, 200)}`,
      false,
    );
  }
  return makeRunnerError(
    PROVIDER_ID,
    "unknown",
    `OpenRouter unexpected status ${status}`,
    false,
  );
}

/**
 * Compute USD cost. For `:free` models the table holds zeros so the
 * result is `0`. For paid entries we use the standard token × $/MTok
 * formula. Returns `undefined` if usage is missing from the response.
 */
function computeOpenRouterCost(
  model: string,
  inputTokens: number | undefined,
  outputTokens: number | undefined,
): number | undefined {
  if (inputTokens === undefined || outputTokens === undefined) return undefined;
  const p = PRICING[model];
  if (p === undefined) return undefined;
  return (
    (inputTokens / 1_000_000) * p.inputUsdPerMtok +
    (outputTokens / 1_000_000) * p.outputUsdPerMtok
  );
}

/**
 * Options for the OpenRouter runner factory.
 */
export interface CreateOpenRouterRunnerOptions {
  apiKey?: string;
  /** Inject a custom `fetch` for tests. */
  fetchFn?: typeof fetch;
  /** Per-call fetch timeout. */
  timeoutMs?: number;
  /** Override the retry policy (tests use this to make retries instant). */
  retry?: {
    maxAttempts?: number;
    baseDelayMs?: number;
    jitterPct?: number;
    sleepFn?: (ms: number) => Promise<void>;
  };
}

/**
 * Construct the OpenRouter runner. Reads `OPENROUTER_API_KEY` from env
 * unless an `apiKey` is supplied explicitly. Throws synchronously if no
 * key is found — fail-fast.
 */
export function createOpenRouterRunner(
  opts: CreateOpenRouterRunnerOptions = {},
): Runner {
  const apiKey = opts.apiKey ?? process.env["OPENROUTER_API_KEY"];
  if (apiKey === undefined || apiKey === "") {
    throw new Error(
      "OpenRouter runner requires OPENROUTER_API_KEY (set env var or pass `apiKey`).",
    );
  }

  const fetchFn = opts.fetchFn ?? fetch;
  const timeoutMs = opts.timeoutMs ?? RUNNER_DEFAULTS.timeoutMs;

  return {
    provider: PROVIDER_ID,
    listModels: () => Object.keys(PRICING),
    async run(req: RunRequest): Promise<RunResponse> {
      if (PRICING[req.model] === undefined) {
        throw makeRunnerError(
          PROVIDER_ID,
          "bad_request",
          `Unsupported OpenRouter model: ${req.model}`,
          false,
        );
      }

      const maxTokens = req.maxTokens ?? RUNNER_DEFAULTS.maxTokens;
      const temperature = req.temperature ?? RUNNER_DEFAULTS.temperature;
      const start = performance.now();

      const respData = (await withRetries(
        async () => {
          const ac = new AbortController();
          const timer = setTimeout(() => ac.abort(), timeoutMs);
          try {
            const r = await fetchFn(ENDPOINT, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                // NEVER log this.
                Authorization: `Bearer ${apiKey}`,
                // OpenRouter-recommended attribution headers (non-secret).
                "HTTP-Referer": ATTRIBUTION_REFERER,
                "X-Title": ATTRIBUTION_TITLE,
              },
              body: JSON.stringify({
                model: req.model,
                messages: [{ role: "user", content: req.prompt }],
                max_tokens: maxTokens,
                temperature,
              }),
              signal: ac.signal,
            });
            if (!r.ok) {
              const body = await r.text();
              throw classifyOpenRouterError(r.status, body, false);
            }
            return (await r.json()) as OpenRouterChatResponse;
          } catch (e) {
            if (
              e instanceof Error &&
              (e.name === "AbortError" || e.message.includes("aborted"))
            ) {
              throw classifyOpenRouterError(0, "", true);
            }
            // Already a RunnerError → rethrow.
            const partial = e as Partial<RunnerError>;
            if (partial?.name === "RunnerError") throw e;
            const msg = e instanceof Error ? e.message : "unknown openrouter error";
            throw makeRunnerError(PROVIDER_ID, "unknown", msg, false);
          } finally {
            clearTimeout(timer);
          }
        },
        {
          maxAttempts: opts.retry?.maxAttempts ?? RUNNER_DEFAULTS.retry.maxAttempts,
          baseDelayMs: opts.retry?.baseDelayMs ?? RUNNER_DEFAULTS.retry.baseDelayMs,
          jitterPct: opts.retry?.jitterPct ?? RUNNER_DEFAULTS.retry.jitterPct,
          sleepFn: opts.retry?.sleepFn,
        },
      )) as OpenRouterChatResponse;

      const latencyMs = performance.now() - start;
      const choice = respData.choices?.[0];
      const output = choice?.message?.content ?? "";
      const inputTokens = respData.usage?.prompt_tokens;
      const outputTokens = respData.usage?.completion_tokens;
      const usdCost = computeOpenRouterCost(req.model, inputTokens, outputTokens);

      return {
        model: req.model,
        output,
        inputTokens,
        outputTokens,
        usdCost,
        latencyMs,
        ts: new Date().toISOString(),
        raw: respData,
      };
    },
  };
}

/** Exported for tests / introspection. Read-only. */
export const __OPENROUTER_PRICING: Readonly<
  Record<string, OpenRouterPricingRow>
> = PRICING;
export const __OPENROUTER_ENDPOINT: string = ENDPOINT;
export const __OPENROUTER_ATTRIBUTION = {
  title: ATTRIBUTION_TITLE,
  referer: ATTRIBUTION_REFERER,
} as const;
