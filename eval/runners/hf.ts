// eval/runners/hf.ts — HuggingFace Inference runner.
//
// Provider: HuggingFace (https://huggingface.co/).
// API:      Serverless Inference API at
//           `https://api-inference.huggingface.co/models/<model>/v1/chat/completions`.
//           This endpoint is OpenAI-compatible for chat models — see
//           https://huggingface.co/docs/inference-providers/en/index — and
//           transparently routes to an Inference Provider when the model
//           is hosted on HF's "Inference Providers" infrastructure.
// Auth:     bearer token from `HF_TOKEN` env var.
//
// Pricing notes:
//   The HF free tier ("Inference API for free tier users") does not
//   charge per token — usage is metered by a monthly request budget.
//   Paid Inference Endpoints prices are tied to the user's deployed
//   hardware (GPU type / hour) and are NOT a function of model id alone.
//   Because we cannot derive a per-token cost from the model id, we
//   report `usdCost: 0` on the free tier. Callers who deploy paid
//   Inference Endpoints should compute cost externally from their HF
//   dashboard.
//
//   The model list below intentionally picks small / permissively-licensed
//   open-weights models that work on the free serverless tier as of 2026-05:
//     - meta-llama/Llama-3.2-1B-Instruct (1B params, Llama license)
//     - meta-llama/Llama-3.2-3B-Instruct (3B params, Llama license)
//     - microsoft/Phi-3.5-mini-instruct  (3.8B params, MIT)
//     - Qwen/Qwen2.5-0.5B-Instruct        (0.5B params, Apache-2.0)
//
// Cold-start handling:
//   HF's serverless inference can return HTTP 503 ("Model is currently
//   loading") for the first request after a model has been evicted from
//   memory. The response body normally includes an `estimated_time`
//   field (seconds). We treat 503-with-loading-body as a *special*
//   retry path that runs OUTSIDE the standard `withRetries` policy:
//   up to two extra cold-start retries at ~30s and ~60s, sleeping at
//   least `estimated_time` between attempts. After the cold-start
//   retries succeed in waking the model, any subsequent transient
//   error falls back to the standard retry policy.
//   See https://huggingface.co/docs/api-inference/detailed_parameters
//   ("Wait for model" section) for the protocol.
//
// Retry (summary):
//   - Cold-start (503+loading or 200+loading): up to 2 extra retries at
//     30s / 60s, then give up.
//   - Standard 429/5xx/timeout: shared `withRetries` policy (3 attempts,
//     1s/2s/4s ± 20%).
//   - Auth/bad_request: terminal, no retry.
//
// Security:
//   `HF_TOKEN` never leaves the runner; provider headers are not echoed.

import { withRetries } from "./_retry.ts";
import {
  makeRunnerError,
  RUNNER_DEFAULTS,
  type Runner,
  type RunnerError,
  type RunRequest,
  type RunResponse,
} from "./_types.ts";

const PROVIDER_ID = "hf";
const ENDPOINT_BASE = "https://api-inference.huggingface.co/models";

/**
 * Model list (free-tier-friendly small open-weights models). HF doesn't
 * meter per-token cost on the free tier so the pricing entry is `0`. We
 * keep the type symmetric with the other runners.
 */
export interface HfPricingRow {
  inputUsdPerMtok: 0;
  outputUsdPerMtok: 0;
}
const FREE: HfPricingRow = { inputUsdPerMtok: 0, outputUsdPerMtok: 0 };

const MODELS: Readonly<Record<string, HfPricingRow>> = {
  "meta-llama/Llama-3.2-1B-Instruct": FREE,
  "meta-llama/Llama-3.2-3B-Instruct": FREE,
  "microsoft/Phi-3.5-mini-instruct": FREE,
  "Qwen/Qwen2.5-0.5B-Instruct": FREE,
};

/**
 * OpenAI-compatible response shape (narrowed). HF chat endpoints return
 * `choices[0].message.content`. The serverless endpoint also returns a
 * special body shape for the "model loading" path with `error` and
 * `estimated_time` fields. Some legacy paths return the loading shape
 * with HTTP 200 instead of 503 — we handle both.
 */
interface HfChatResponse {
  choices?: Array<{
    message?: { content?: string | null };
    finish_reason?: string;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
  };
  error?: string;
  estimated_time?: number;
}

/** Default cold-start retry schedule (ms). Tests inject shorter values. */
const DEFAULT_COLD_START_DELAYS_MS: readonly number[] = [30_000, 60_000];
const MAX_COLD_START_DELAY_MS = 60_000;

/** Sleep helper used by the cold-start loop (the standard retry loop has
 * its own sleep via withRetries). */
function defaultSleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Classify a non-loading HTTP error into a `RunnerError`. Mirrors the
 * Anthropic / Groq / Together policy so all runners look identical to
 * the build orchestrator.
 */
function classifyHfError(
  status: number,
  body: string,
  isAbort: boolean,
): RunnerError {
  if (isAbort) {
    return makeRunnerError(
      PROVIDER_ID,
      "timeout",
      "HuggingFace request timed out",
      true,
    );
  }
  if (status === 401 || status === 403) {
    return makeRunnerError(
      PROVIDER_ID,
      "auth",
      `HuggingFace auth failed (${status})`,
      false,
    );
  }
  if (status === 429) {
    return makeRunnerError(
      PROVIDER_ID,
      "rate_limit",
      `HuggingFace rate limit (429): ${body.slice(0, 200)}`,
      true,
    );
  }
  if (status >= 500) {
    return makeRunnerError(
      PROVIDER_ID,
      "server",
      `HuggingFace server error (${status}): ${body.slice(0, 200)}`,
      true,
    );
  }
  if (status >= 400) {
    return makeRunnerError(
      PROVIDER_ID,
      "bad_request",
      `HuggingFace bad request (${status}): ${body.slice(0, 200)}`,
      false,
    );
  }
  return makeRunnerError(
    PROVIDER_ID,
    "unknown",
    `HuggingFace unexpected status ${status}`,
    false,
  );
}

/**
 * Detect HF's "model is loading" cold-start response. HF returns 503
 * with `{"error":"Model X is currently loading","estimated_time":42.3}`.
 * Some legacy paths return 200 with the same body — we handle both.
 *
 * Returns the recommended wait in ms (bounded by `MAX_COLD_START_DELAY_MS`)
 * when this looks like a loading event, else `null`.
 */
function coldStartWaitMs(status: number, body: HfChatResponse): number | null {
  const looksLikeLoading =
    typeof body.error === "string" && /currently loading/i.test(body.error);
  if (status !== 503 && !looksLikeLoading) return null;
  const estSec =
    typeof body.estimated_time === "number" ? body.estimated_time : 30;
  return Math.min(
    MAX_COLD_START_DELAY_MS,
    Math.max(0, Math.round(estSec * 1000)),
  );
}

/**
 * One fetch attempt against the HF endpoint. Returns parsed body and the
 * HTTP status. Errors (network / abort) propagate as exceptions.
 */
async function callHfOnce(
  apiKey: string,
  req: RunRequest,
  fetchFn: typeof fetch,
  timeoutMs: number,
): Promise<{ status: number; body: HfChatResponse; rawText: string }> {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  try {
    const url = `${ENDPOINT_BASE}/${req.model}/v1/chat/completions`;
    const res = await fetchFn(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // NEVER log this.
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: req.model,
        messages: [{ role: "user", content: req.prompt }],
        max_tokens: req.maxTokens ?? RUNNER_DEFAULTS.maxTokens,
        temperature: req.temperature ?? RUNNER_DEFAULTS.temperature,
      }),
      signal: ac.signal,
    });
    const rawText = await res.text();
    let body: HfChatResponse;
    try {
      body =
        rawText.length > 0 ? (JSON.parse(rawText) as HfChatResponse) : {};
    } catch {
      // Non-JSON body (e.g. plain-text 5xx error page). Treat as empty;
      // the status code drives error classification anyway.
      body = {};
    }
    return { status: res.status, body, rawText };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Options for the HuggingFace runner factory.
 */
export interface CreateHfRunnerOptions {
  apiKey?: string;
  /** Inject a custom `fetch` for tests. */
  fetchFn?: typeof fetch;
  /** Per-call fetch timeout. */
  timeoutMs?: number;
  /**
   * Override the cold-start delay schedule. Tests use short values
   * (e.g. `[5, 10]`); production should leave this undefined.
   */
  coldStartDelaysMs?: readonly number[];
  /** Sleep override for the cold-start loop (tests pass a no-op). */
  coldStartSleepFn?: (ms: number) => Promise<void>;
  /** Override the standard retry policy. */
  retry?: {
    maxAttempts?: number;
    baseDelayMs?: number;
    jitterPct?: number;
    sleepFn?: (ms: number) => Promise<void>;
  };
}

/**
 * Construct the HuggingFace runner. Reads `HF_TOKEN` from env unless an
 * `apiKey` is supplied explicitly. Throws synchronously if no key is
 * found — fail-fast.
 */
export function createHfRunner(opts: CreateHfRunnerOptions = {}): Runner {
  const apiKey = opts.apiKey ?? process.env["HF_TOKEN"];
  if (apiKey === undefined || apiKey === "") {
    throw new Error(
      "HuggingFace runner requires HF_TOKEN (set env var or pass `apiKey`).",
    );
  }

  const fetchFn = opts.fetchFn ?? fetch;
  const timeoutMs = opts.timeoutMs ?? RUNNER_DEFAULTS.timeoutMs;
  const coldDelays = opts.coldStartDelaysMs ?? DEFAULT_COLD_START_DELAYS_MS;
  const coldSleep = opts.coldStartSleepFn ?? defaultSleep;

  return {
    provider: PROVIDER_ID,
    listModels: () => Object.keys(MODELS),
    async run(req: RunRequest): Promise<RunResponse> {
      if (MODELS[req.model] === undefined) {
        throw makeRunnerError(
          PROVIDER_ID,
          "bad_request",
          `Unsupported HuggingFace model: ${req.model}`,
          false,
        );
      }

      const start = performance.now();

      // Inner attempt: one call + cold-start handling. We wrap this in
      // `withRetries` so 429 / 5xx / timeout get the shared policy too.
      // Cold-start retries live INSIDE this closure so they don't burn
      // the standard retry budget.
      const respData = (await withRetries(
        async () => {
          let coldStartIdx = 0;
          for (;;) {
            let res: { status: number; body: HfChatResponse; rawText: string };
            try {
              res = await callHfOnce(apiKey, req, fetchFn, timeoutMs);
            } catch (e) {
              if (
                e instanceof Error &&
                (e.name === "AbortError" || e.message.includes("aborted"))
              ) {
                throw classifyHfError(0, "", true);
              }
              const msg = e instanceof Error ? e.message : "unknown hf error";
              throw makeRunnerError(PROVIDER_ID, "unknown", msg, false);
            }

            const csWait = coldStartWaitMs(res.status, res.body);
            if (csWait !== null) {
              const sched = coldDelays[coldStartIdx];
              if (coldStartIdx < coldDelays.length && sched !== undefined) {
                const wait = Math.max(csWait, sched);
                coldStartIdx += 1;
                await coldSleep(wait);
                continue; // re-issue the request after waiting
              }
              // Exceeded cold-start retries; surface as retryable server
              // error so the standard policy can take one more swing.
              throw makeRunnerError(
                PROVIDER_ID,
                "server",
                `HuggingFace model still loading after ${coldDelays.length} cold-start retries`,
                true,
              );
            }

            if (res.status >= 200 && res.status < 300) {
              return res.body;
            }

            throw classifyHfError(res.status, res.rawText, false);
          }
        },
        {
          maxAttempts: opts.retry?.maxAttempts ?? RUNNER_DEFAULTS.retry.maxAttempts,
          baseDelayMs: opts.retry?.baseDelayMs ?? RUNNER_DEFAULTS.retry.baseDelayMs,
          jitterPct: opts.retry?.jitterPct ?? RUNNER_DEFAULTS.retry.jitterPct,
          sleepFn: opts.retry?.sleepFn,
        },
      )) as HfChatResponse;

      const latencyMs = performance.now() - start;
      const choice = respData.choices?.[0];
      const output = choice?.message?.content ?? "";
      const inputTokens = respData.usage?.prompt_tokens;
      const outputTokens = respData.usage?.completion_tokens;
      // Free tier → usdCost = 0. Paid Inference Endpoints cost depends
      // on the user's deployed hardware; the runner is not positioned to
      // compute that.
      const usdCost = 0;

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
export const __HF_MODELS: Readonly<Record<string, HfPricingRow>> = MODELS;
export const __HF_DEFAULT_COLD_START_DELAYS_MS: readonly number[] =
  DEFAULT_COLD_START_DELAYS_MS;
export const __HF_ENDPOINT_BASE: string = ENDPOINT_BASE;
