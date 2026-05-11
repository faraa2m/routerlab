// eval/runners/together.ts — Together AI runner.
//
// Provider: Together AI (https://api.together.xyz/).
// API:      OpenAI-compatible REST at `/v1/chat/completions`.
// Auth:     bearer token from `TOGETHER_API_KEY` env var.
//
// Pricing notes:
//   Together publishes per-model token pricing at
//   https://www.together.ai/pricing. The four open-weights models we
//   route across here run on Together's hosted endpoints; pricing below
//   mirrors the public catalog as of 2026-05 and is a worst-case (catalog)
//   figure. If the calling user has free credits / a free trial those
//   are deducted server-side; we still report the catalog price for
//   cost-honesty in the audit log. Refresh via `scripts/refresh-pricing`
//   (Phase 3) when Together rev's its rates.
//
// Cold-start:
//   Together's hosted open-weights endpoints are warm-pooled. Unlike HF
//   Inference there is no documented cold-start 503; we apply the shared
//   exponential-backoff retry policy (rate_limit/timeout/server) only.
//
// Retry:
//   Shared `withRetries` helper from `_retry.ts` — up to 3 attempts with
//   jittered exponential backoff (1s, 2s, 4s ± 20%). 4xx (other than 429)
//   is terminal; 429/5xx is retried.
//
// Security:
//   Never log the bearer token. The token is read once at construction
//   and held in a closure; it is never serialized into `RunResponse.raw`.
//   Provider response bodies are stored verbatim in `raw` for the audit
//   log but the runner does not echo request headers.

import { withRetries } from "./_retry.ts";
import {
  makeRunnerError,
  RUNNER_DEFAULTS,
  type Runner,
  type RunnerError,
  type RunRequest,
  type RunResponse,
} from "./_types.ts";

const PROVIDER_ID = "together";
const ENDPOINT = "https://api.together.xyz/v1/chat/completions";

/**
 * Per-million-token pricing (USD). Numbers are catalog rates from
 * Together's public pricing page as of 2026-05. Together's open-weights
 * pricing is typically symmetric across input/output for the same model,
 * so we record both fields explicitly to keep the type symmetric with
 * the other runners.
 */
export interface TogetherPricingRow {
  inputUsdPerMtok: number;
  outputUsdPerMtok: number;
}

const PRICING: Readonly<Record<string, TogetherPricingRow>> = {
  "meta-llama/Llama-3.3-70B-Instruct-Turbo": {
    inputUsdPerMtok: 0.88,
    outputUsdPerMtok: 0.88,
  },
  "meta-llama/Llama-3.1-8B-Instruct": {
    inputUsdPerMtok: 0.18,
    outputUsdPerMtok: 0.18,
  },
  "mistralai/Mixtral-8x7B-Instruct-v0.1": {
    inputUsdPerMtok: 0.6,
    outputUsdPerMtok: 0.6,
  },
  "Qwen/Qwen2.5-7B-Instruct-Turbo": {
    inputUsdPerMtok: 0.3,
    outputUsdPerMtok: 0.3,
  },
};

/**
 * OpenAI-compatible chat-completions response (narrowed to the fields we
 * read). Together returns provider-specific extras under `usage` but the
 * core shape matches OpenAI's.
 */
interface TogetherChatResponse {
  choices?: Array<{
    message?: { content?: string | null };
    finish_reason?: string;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
  };
  error?: { message?: string; type?: string };
}

/**
 * Map an HTTP status code (and optional body) to a `RunnerError`. Mirrors
 * the policy used by the Anthropic / Groq runners so the build orchestrator
 * sees identical error shapes regardless of provider.
 */
function classifyTogetherError(
  status: number,
  body: string,
  isAbort: boolean,
): RunnerError {
  if (isAbort) {
    return makeRunnerError(
      PROVIDER_ID,
      "timeout",
      "Together request timed out",
      true,
    );
  }
  if (status === 401 || status === 403) {
    return makeRunnerError(
      PROVIDER_ID,
      "auth",
      `Together auth failed (${status})`,
      false,
    );
  }
  if (status === 429) {
    return makeRunnerError(
      PROVIDER_ID,
      "rate_limit",
      `Together rate limit (429): ${body.slice(0, 200)}`,
      true,
    );
  }
  if (status >= 500) {
    return makeRunnerError(
      PROVIDER_ID,
      "server",
      `Together server error (${status}): ${body.slice(0, 200)}`,
      true,
    );
  }
  if (status >= 400) {
    return makeRunnerError(
      PROVIDER_ID,
      "bad_request",
      `Together bad request (${status}): ${body.slice(0, 200)}`,
      false,
    );
  }
  return makeRunnerError(
    PROVIDER_ID,
    "unknown",
    `Together unexpected status ${status}`,
    false,
  );
}

/**
 * Compute USD cost from a Together response's `usage` field, using the
 * per-model pricing table. Returns `undefined` if usage is missing or the
 * model is not in `PRICING` — the runner emits an uncosted `RunResponse`
 * rather than throwing in that case (the audit log will record the gap).
 */
function computeTogetherCost(
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
 * Options for the Together runner factory.
 */
export interface CreateTogetherRunnerOptions {
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
 * Construct the Together runner. Reads `TOGETHER_API_KEY` from env unless
 * an `apiKey` is supplied explicitly. Throws synchronously if no key is
 * found — consistent with the Anthropic / Groq runners' fail-fast
 * behavior, which keeps the build orchestrator from discovering missing creds
 * mid-eval.
 */
export function createTogetherRunner(
  opts: CreateTogetherRunnerOptions = {},
): Runner {
  const apiKey = opts.apiKey ?? process.env["TOGETHER_API_KEY"];
  if (apiKey === undefined || apiKey === "") {
    throw new Error(
      "Together runner requires TOGETHER_API_KEY (set env var or pass `apiKey`).",
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
          `Unsupported Together model: ${req.model}`,
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
              throw classifyTogetherError(r.status, body, false);
            }
            return (await r.json()) as TogetherChatResponse;
          } catch (e) {
            if (
              e instanceof Error &&
              (e.name === "AbortError" || e.message.includes("aborted"))
            ) {
              throw classifyTogetherError(0, "", true);
            }
            // Already a RunnerError → rethrow so withRetries sees it.
            const partial = e as Partial<RunnerError>;
            if (partial?.name === "RunnerError") throw e;
            const msg = e instanceof Error ? e.message : "unknown together error";
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
      )) as TogetherChatResponse;

      const latencyMs = performance.now() - start;
      const choice = respData.choices?.[0];
      const output = choice?.message?.content ?? "";
      const inputTokens = respData.usage?.prompt_tokens;
      const outputTokens = respData.usage?.completion_tokens;
      const usdCost = computeTogetherCost(req.model, inputTokens, outputTokens);

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
export const __TOGETHER_PRICING: Readonly<Record<string, TogetherPricingRow>> =
  PRICING;
export const __TOGETHER_ENDPOINT: string = ENDPOINT;
