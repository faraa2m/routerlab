// eval/runners/anthropic.ts — Anthropic runner for `eval/runners`.
//
// Uses `@anthropic-ai/sdk`'s `messages.create` to call Claude models.
// Cost computation pulls from `_pricing.ts` (which mirrors
// `packages/core/src/candidates.json`). Retries are handled by the shared
// `withRetries` helper; 429 / 5xx / timeout are retryable, 4xx (auth /
// bad_request) is not.
//
// The SDK is permitted by the brief (it's a thin wrapper that gives us
// typed `usage` fields). We still hand-roll the retry loop because the
// SDK's built-in retry doesn't give us the error classification we need
// for the build orchestrator.

import Anthropic from "@anthropic-ai/sdk";
import { computeUsdCost } from "./_pricing.ts";
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
 * Canonical Anthropic model ids supported by this runner. These match
 * `packages/core/src/candidates.json` and are passed through as-is to the
 * Anthropic SDK (Anthropic accepts these short ids in 2026's API).
 */
const ANTHROPIC_MODELS: readonly string[] = [
  "claude-opus-4-7",
  "claude-sonnet-4-6",
  "claude-haiku-4-5",
] as const;

/**
 * Anthropic SDK errors expose `status` (HTTP code) and sometimes `error`.
 * We narrow the unknown thrown value into this shape so the classifier
 * below stays strict-mode clean.
 */
interface AnthropicLikeError {
  status?: number;
  message?: string;
  name?: string;
}

function isAnthropicLikeError(e: unknown): e is AnthropicLikeError {
  return typeof e === "object" && e !== null && ("status" in e || "message" in e);
}

/**
 * Map an SDK / fetch error to a typed `RunnerError`. The brief specifies
 * the mapping: 401 → auth (not retryable), 429 → rate_limit (retryable),
 * ≥500 → server (retryable), AbortError → timeout (retryable), else
 * unknown (not retryable so we don't loop forever on weird states).
 */
function classifyAnthropicError(e: unknown): RunnerError {
  // AbortController fires DOMException with name "AbortError". The Bun /
  // Node difference is irrelevant here — we just key on the name string.
  const isAbort =
    e instanceof Error && (e.name === "AbortError" || e.message.includes("aborted"));
  if (isAbort) {
    return makeRunnerError("anthropic", "timeout", "Anthropic request timed out", true);
  }

  if (isAnthropicLikeError(e)) {
    const status = e.status ?? 0;
    const msg = e.message ?? "Anthropic request failed";
    if (status === 401 || status === 403) {
      return makeRunnerError("anthropic", "auth", msg, false);
    }
    if (status === 429) {
      return makeRunnerError("anthropic", "rate_limit", msg, true);
    }
    if (status >= 500) {
      return makeRunnerError("anthropic", "server", msg, true);
    }
    if (status >= 400) {
      return makeRunnerError("anthropic", "bad_request", msg, false);
    }
  }

  const fallbackMsg =
    e instanceof Error ? e.message : "Unknown error from Anthropic runner";
  return makeRunnerError("anthropic", "unknown", fallbackMsg, false);
}

/**
 * Minimal shape we need from the Anthropic SDK response. Kept local so we
 * don't depend on internal SDK types.
 */
interface AnthropicMessageResponse {
  content: Array<{ type: string; text?: string }>;
  usage?: { input_tokens?: number; output_tokens?: number };
  stop_reason?: string;
}

/**
 * Concatenate `text` parts of the response. Anthropic returns a content
 * array (tool use, text, etc.); for the runner we only emit the text.
 */
function extractText(resp: AnthropicMessageResponse): string {
  const parts: string[] = [];
  for (const block of resp.content) {
    if (block.type === "text" && typeof block.text === "string") {
      parts.push(block.text);
    }
  }
  return parts.join("");
}

/**
 * Options accepted by the runner factory. The SDK client is injectable so
 * tests can pass a stub without touching env vars.
 */
export interface CreateAnthropicRunnerOptions {
  apiKey?: string;
  /** Inject a custom SDK client for tests. */
  client?: Pick<Anthropic, "messages">;
  /** Override the per-call fetch timeout. */
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
 * Construct the Anthropic runner. Reads `ANTHROPIC_API_KEY` from env unless
 * an `apiKey` is supplied explicitly. Throws synchronously if no key is
 * found AND no `client` is injected — fail-fast keeps the build orchestrator
 * from discovering missing creds mid-eval.
 */
export function createAnthropicRunner(
  opts: CreateAnthropicRunnerOptions = {},
): Runner {
  const apiKey = opts.apiKey ?? process.env["ANTHROPIC_API_KEY"];
  // We allow a stub `client` without an apiKey, which is what mocked tests do.
  if (opts.client === undefined && (apiKey === undefined || apiKey === "")) {
    throw new Error(
      "Anthropic runner requires ANTHROPIC_API_KEY (set env var or pass `apiKey`).",
    );
  }

  const client: Pick<Anthropic, "messages"> =
    opts.client ?? new Anthropic({ apiKey: apiKey as string });

  const timeoutMs = opts.timeoutMs ?? RUNNER_DEFAULTS.timeoutMs;

  return {
    provider: "anthropic",
    listModels: () => [...ANTHROPIC_MODELS],
    async run(req: RunRequest): Promise<RunResponse> {
      if (!ANTHROPIC_MODELS.includes(req.model)) {
        throw makeRunnerError(
          "anthropic",
          "bad_request",
          `Unsupported Anthropic model: ${req.model}`,
          false,
        );
      }

      const maxTokens = req.maxTokens ?? RUNNER_DEFAULTS.maxTokens;
      const temperature = req.temperature ?? RUNNER_DEFAULTS.temperature;

      const start = performance.now();
      // The abort controller fires on `timeoutMs` and is passed to the SDK
      // via its options bag — the SDK propagates the signal to its fetch.
      const ac = new AbortController();
      const timer = setTimeout(() => ac.abort(), timeoutMs);

      try {
        const resp = (await withRetries(
          async () => {
            try {
              return (await client.messages.create(
                {
                  model: req.model,
                  max_tokens: maxTokens,
                  temperature,
                  messages: [{ role: "user", content: req.prompt }],
                },
                { signal: ac.signal },
              )) as unknown as AnthropicMessageResponse;
            } catch (e) {
              throw classifyAnthropicError(e);
            }
          },
          {
            maxAttempts: opts.retry?.maxAttempts ?? RUNNER_DEFAULTS.retry.maxAttempts,
            baseDelayMs: opts.retry?.baseDelayMs ?? RUNNER_DEFAULTS.retry.baseDelayMs,
            jitterPct: opts.retry?.jitterPct ?? RUNNER_DEFAULTS.retry.jitterPct,
            sleepFn: opts.retry?.sleepFn,
          },
        )) as AnthropicMessageResponse;

        const latencyMs = performance.now() - start;
        const inputTokens = resp.usage?.input_tokens;
        const outputTokens = resp.usage?.output_tokens;
        const usdCost =
          inputTokens !== undefined && outputTokens !== undefined
            ? computeUsdCost(req.model, inputTokens, outputTokens)
            : undefined;

        return {
          model: req.model,
          output: extractText(resp),
          inputTokens,
          outputTokens,
          usdCost,
          latencyMs,
          ts: new Date().toISOString(),
          raw: resp,
        };
      } finally {
        clearTimeout(timer);
      }
    },
  };
}

/**
 * Exported for tests and introspection. The list is read-only.
 */
export const __ANTHROPIC_MODELS: readonly string[] = ANTHROPIC_MODELS;
