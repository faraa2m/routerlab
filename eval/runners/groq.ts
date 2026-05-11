// eval/runners/groq.ts — Groq runner.
//
// Groq exposes an OpenAI-compatible REST API at
// `https://api.groq.com/openai/v1/chat/completions`. We hit it with Bun's
// built-in `fetch` (no SDK dependency — keeps install surface small).
//
// Pricing: Groq's developer tier is genuinely free. The runner reports
// `usdCost: 0` for free-tier models (see `GROQ_FREE_TIER_MODELS` in
// `_pricing.ts`). Paid-tier reference numbers live in `candidates.json`
// and are surfaced here only as a fallback — the brief asks us to mark
// usdCost: 0 for free-tier responses. If a future paid tier is wired up,
// flip the `paidMode` flag at runner construction.
//
// Free-tier rate limits are real and we will hit them under any sustained
// eval load. The shared `withRetries` helper handles the backoff.

import { GROQ_FREE_TIER_MODELS, GROQ_MODEL_ALIASES, computeUsdCost } from "./_pricing.ts";
import { withRetries } from "./_retry.ts";
import {
  makeRunnerError,
  RUNNER_DEFAULTS,
  type RunRequest,
  type RunResponse,
  type Runner,
  type RunnerError,
} from "./_types.ts";

/**
 * The canonical Groq model ids accepted as input. We accept both the
 * "long" Groq-API form (`llama-3.3-70b-versatile`) and the short form
 * used in `candidates.json` (`llama-3.3-70b`); see `GROQ_MODEL_ALIASES`.
 *
 * NOTE: verify these against
 * https://console.groq.com/docs/models — the long-form names match Groq's
 * 2026 published model list.
 */
const GROQ_MODELS_CANONICAL: readonly string[] = [
  "llama-3.3-70b-versatile",
  "llama-3.1-8b-instant",
  "mixtral-8x7b-32768",
] as const;

/**
 * Convenience: union of long-form + short-form model ids the runner accepts.
 * The factory's `listModels()` returns the long-form (Groq's canonical
 * IDs) — the short forms are accepted as alias inputs but normalize back
 * to the long form before the HTTP call.
 */
const GROQ_MODELS_ACCEPTED: readonly string[] = [
  ...GROQ_MODELS_CANONICAL,
  ...Object.values(GROQ_MODEL_ALIASES),
];

/**
 * The OpenAI-compatible chat completion endpoint Groq serves at.
 */
const GROQ_CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions";

/**
 * Subset of the OpenAI-compatible response we need. The `choices` array
 * holds the generated text; `usage` reports token counts.
 */
interface GroqChatCompletionResponse {
  choices: Array<{
    index?: number;
    message?: { role?: string; content?: string };
    finish_reason?: string;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

/**
 * Map an HTTP status / abort / unknown error to a typed `RunnerError`.
 * Same mapping policy as the Anthropic runner — keep these in sync so
 * the build orchestrator can treat all runners uniformly.
 */
function classifyGroqError(status: number, body: string, isAbort: boolean): RunnerError {
  if (isAbort) {
    return makeRunnerError("groq", "timeout", "Groq request timed out", true);
  }
  if (status === 401 || status === 403) {
    return makeRunnerError("groq", "auth", `Groq auth failed (${status})`, false);
  }
  if (status === 429) {
    return makeRunnerError("groq", "rate_limit", `Groq rate limit (429): ${body}`, true);
  }
  if (status >= 500) {
    return makeRunnerError("groq", "server", `Groq server error (${status}): ${body}`, true);
  }
  if (status >= 400) {
    return makeRunnerError(
      "groq",
      "bad_request",
      `Groq bad request (${status}): ${body}`,
      false,
    );
  }
  return makeRunnerError("groq", "unknown", `Groq unknown error (${status}): ${body}`, false);
}

/**
 * Options for the Groq runner factory.
 */
export interface CreateGroqRunnerOptions {
  apiKey?: string;
  /** Inject a custom `fetch` for tests. */
  fetchFn?: typeof fetch;
  /** Per-call fetch timeout. */
  timeoutMs?: number;
  /**
   * When `true`, compute cost from `candidates.json` paid-tier pricing
   * instead of reporting `0`. Default is `false` (free tier).
   */
  paidMode?: boolean;
  /** Override the retry policy (tests use this to make retries instant). */
  retry?: {
    maxAttempts?: number;
    baseDelayMs?: number;
    jitterPct?: number;
    sleepFn?: (ms: number) => Promise<void>;
  };
}

/**
 * Construct the Groq runner. Reads `GROQ_API_KEY` from env unless an
 * `apiKey` is supplied explicitly. Throws synchronously if no key is
 * found (consistent with the Anthropic runner's fail-fast behavior).
 */
export function createGroqRunner(opts: CreateGroqRunnerOptions = {}): Runner {
  const apiKey = opts.apiKey ?? process.env["GROQ_API_KEY"];
  if (apiKey === undefined || apiKey === "") {
    throw new Error(
      "Groq runner requires GROQ_API_KEY (set env var or pass `apiKey`).",
    );
  }

  const fetchFn = opts.fetchFn ?? fetch;
  const timeoutMs = opts.timeoutMs ?? RUNNER_DEFAULTS.timeoutMs;
  const paidMode = opts.paidMode ?? false;

  return {
    provider: "groq",
    // We expose the long-form canonical IDs (Groq's API expects these).
    listModels: () => [...GROQ_MODELS_CANONICAL],
    async run(req: RunRequest): Promise<RunResponse> {
      if (!GROQ_MODELS_ACCEPTED.includes(req.model)) {
        throw makeRunnerError(
          "groq",
          "bad_request",
          `Unsupported Groq model: ${req.model}`,
          false,
        );
      }

      // Normalize short-form id (`llama-3.3-70b`) → long-form (`llama-3.3-70b-versatile`).
      // Groq's API wants the long form; our cost table is keyed on short form.
      const groqApiModel = (() => {
        // If req.model is already long-form, pass through; else look up alias.
        if (GROQ_MODELS_CANONICAL.includes(req.model)) return req.model;
        // Short form: find which long-form maps to this short form.
        for (const [longForm, shortForm] of Object.entries(GROQ_MODEL_ALIASES)) {
          if (shortForm === req.model) return longForm;
        }
        // Defensive: shouldn't reach here given the includes() above.
        return req.model;
      })();

      // Short-form id (matches `candidates.json` keys). Used both as the
      // returned `RunResponse.model` and the pricing-table key.
      const shortFormModel =
        GROQ_MODEL_ALIASES[groqApiModel] ?? groqApiModel;

      const maxTokens = req.maxTokens ?? RUNNER_DEFAULTS.maxTokens;
      const temperature = req.temperature ?? RUNNER_DEFAULTS.temperature;

      const start = performance.now();

      const respData = (await withRetries(
        async () => {
          const ac = new AbortController();
          const timer = setTimeout(() => ac.abort(), timeoutMs);
          let isAbort = false;
          try {
            const r = await fetchFn(GROQ_CHAT_ENDPOINT, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                // NEVER log this — keep it out of stdout / stderr.
                Authorization: `Bearer ${apiKey}`,
              },
              body: JSON.stringify({
                model: groqApiModel,
                messages: [{ role: "user", content: req.prompt }],
                max_tokens: maxTokens,
                temperature,
              }),
              signal: ac.signal,
            });
            if (!r.ok) {
              const body = await r.text();
              throw classifyGroqError(r.status, body, false);
            }
            return (await r.json()) as GroqChatCompletionResponse;
          } catch (e) {
            if (e instanceof Error && (e.name === "AbortError" || e.message.includes("aborted"))) {
              isAbort = true;
              throw classifyGroqError(0, "", isAbort);
            }
            // Already a RunnerError → rethrow as-is so withRetries sees it.
            const partial = e as Partial<RunnerError>;
            if (partial?.name === "RunnerError") throw e;
            // Otherwise this is some unexpected fetch/runtime failure.
            const msg = e instanceof Error ? e.message : "unknown groq error";
            throw makeRunnerError("groq", "unknown", msg, false);
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
      )) as GroqChatCompletionResponse;

      const latencyMs = performance.now() - start;
      const choice = respData.choices[0];
      const output = choice?.message?.content ?? "";
      const inputTokens = respData.usage?.prompt_tokens;
      const outputTokens = respData.usage?.completion_tokens;

      // Free-tier path: usdCost = 0. Paid-tier path: pull from candidates.json.
      const isFreeTier =
        !paidMode &&
        (GROQ_FREE_TIER_MODELS.has(groqApiModel) || GROQ_FREE_TIER_MODELS.has(shortFormModel));
      const usdCost = (() => {
        if (isFreeTier) return 0;
        if (inputTokens === undefined || outputTokens === undefined) return undefined;
        return computeUsdCost(shortFormModel, inputTokens, outputTokens);
      })();

      return {
        // Always return the short-form id — this is the canonical key used
        // throughout routerlab (router engine, candidates.json, etc.).
        model: shortFormModel,
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

/** Exported for tests + introspection. Read-only. */
export const __GROQ_MODELS_CANONICAL: readonly string[] = GROQ_MODELS_CANONICAL;
export const __GROQ_CHAT_ENDPOINT: string = GROQ_CHAT_ENDPOINT;
