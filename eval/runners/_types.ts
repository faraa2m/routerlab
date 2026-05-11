// eval/runners/_types.ts — shared runner contract.
//
// Every per-provider runner under `eval/runners/<provider>.ts` implements
// the `Runner` interface defined here. The orchestrator (Phase 3) treats
// runners as black boxes that take a `RunRequest` and return a
// `RunResponse` or throw a `RunnerError`. Keeping this contract narrow and
// dependency-free is what lets us swap providers, add new ones, and
// mock-test individual runners in isolation.
//
// This file is co-owned by `router-runners-A` (anthropic + groq) and
// `router-runners-B` (together + hf + openrouter). Whichever agent lands
// first creates it; the second must not reshape it without coordination.

/**
 * What a caller passes in to a runner. Model identifiers must be exact
 * canonical strings as documented in `packages/core/src/candidates.json`
 * — runners are not in the business of fuzzy matching model names.
 */
export interface RunRequest {
  /** Canonical model id (matches an entry in `candidates.json`). */
  model: string;
  prompt: string;
  /** Provider-side max output tokens. Defaults are per-runner. */
  maxTokens?: number;
  /** Sampling temperature. Defaults are per-runner. */
  temperature?: number;
}

/**
 * The successful return value of `Runner.run()`. Every field is informational
 * except `output`, which is the model's generated text. The rest exists so
 * downstream code (router engine, eval harness, audit log) can reason about
 * cost, latency, and reproducibility without re-calling the provider.
 *
 * Pricing notes:
 *   - `usdCost` is computed by the runner from a per-model pricing table
 *     mirroring `packages/core/src/candidates.json`. If a provider runs on a
 *     genuinely free tier (e.g. Groq free tier), the runner returns
 *     `usdCost: 0` and documents that in its file header.
 *   - Token counts come from the provider's own `usage` field where
 *     available; when a provider does not report usage, the runner leaves
 *     `inputTokens` / `outputTokens` undefined and `usdCost` undefined.
 */
export interface RunResponse {
  /** The canonical model id from the request, echoed back. */
  model: string;
  /** The generated text. */
  output: string;
  inputTokens?: number;
  outputTokens?: number;
  /** Computed cost in USD (see file header for pricing-table notes). */
  usdCost?: number;
  /** Wall-clock latency in ms, measured around the provider call. */
  latencyMs: number;
  /** ISO-8601 timestamp at request completion. */
  ts: string;
  /** Raw provider response body, retained for the audit log. */
  raw?: unknown;
}

/**
 * Classified provider error. Runners normalize HTTP status codes and
 * exception types into one of the `reason` values so the build orchestrator can
 * decide whether to retry, fall back to a different provider, or surface
 * the error to the caller.
 *
 * Retry semantics:
 *   - `retryable: true`  → the build orchestrator may retry the same provider.
 *     Individual runners are also expected to retry internally with
 *     exponential backoff for `rate_limit`, `timeout`, and `server` before
 *     giving up and throwing.
 *   - `retryable: false` → terminal; do not retry the same provider on
 *     this request. Caller should fall back to a different candidate.
 */
export interface RunnerError extends Error {
  name: "RunnerError";
  provider: string;
  reason: "auth" | "rate_limit" | "timeout" | "bad_request" | "server" | "unknown";
  retryable: boolean;
}

/**
 * The runner contract every provider implementation satisfies.
 *
 * `listModels()` returns the exact set of canonical model ids this runner
 * supports — used by the factory to validate `RunRequest.model` before
 * dispatch.
 */
export interface Runner {
  /** Provider identifier, e.g. "anthropic", "groq". */
  provider: string;
  /** Canonical model ids this runner can serve. */
  listModels(): string[];
  /** Execute one request against the provider. */
  run(req: RunRequest): Promise<RunResponse>;
}

/**
 * Construct a typed `RunnerError`. Centralized so all runners produce
 * structurally identical errors and so we don't accidentally drop a field.
 */
export function makeRunnerError(
  provider: string,
  reason: RunnerError["reason"],
  message: string,
  retryable: boolean,
): RunnerError {
  const err = new Error(message) as RunnerError;
  err.name = "RunnerError";
  err.provider = provider;
  err.reason = reason;
  err.retryable = retryable;
  return err;
}

/**
 * Common defaults used by every runner unless a `RunRequest` overrides them.
 *
 * Keep these conservative — `maxTokens` is small enough to be cheap during
 * eval runs but large enough for the common router task classes (QA,
 * classification, summarization). Codegen / reasoning tasks should pass an
 * explicit larger `maxTokens`.
 */
export const RUNNER_DEFAULTS = {
  maxTokens: 1024,
  temperature: 0.2,
  /** Per-call fetch timeout in ms. */
  timeoutMs: 60_000,
  /** Retry policy shared across all runners. */
  retry: {
    maxAttempts: 3,
    baseDelayMs: 1_000,
    jitterPct: 0.2,
  },
} as const;
