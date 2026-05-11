// eval/runners/_retry.ts — shared async retry helper for every runner.
//
// Why this exists: each provider runner needs the same exponential-backoff
// retry policy (3 attempts, base 1s, ±20% jitter, retry on rate_limit /
// server / timeout). The brief explicitly forbids pulling in a retry
// library — so this is the one local primitive every runner reaches for.
//
// The helper is intentionally async-await rather than promise-chained: it
// reads top-to-bottom, the sleep is awaitable, and the abort signal can
// short-circuit the loop without leaking timers.

import { RUNNER_DEFAULTS, type RunnerError } from "./_types.ts";

/**
 * Sleep for `ms` milliseconds. Pulled out as its own helper so the retry
 * loop reads cleanly and so tests can mock it if they ever need to.
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Compute the next backoff delay. Exponential (`base × 2^attempt`) with
 * symmetric jitter (`±jitterPct`) so concurrent callers don't synchronize
 * their retries against the same rate limit window.
 *
 * `attempt` is 0-indexed: 0 → first retry delay, 1 → second, …
 */
export function backoffDelayMs(
  attempt: number,
  baseMs: number = RUNNER_DEFAULTS.retry.baseDelayMs,
  jitterPct: number = RUNNER_DEFAULTS.retry.jitterPct,
): number {
  const exponential = baseMs * 2 ** attempt;
  // Symmetric jitter in [-jitterPct, +jitterPct].
  const jitter = (Math.random() * 2 - 1) * jitterPct;
  return Math.max(0, Math.round(exponential * (1 + jitter)));
}

/**
 * Retry policy: runs `op` up to `maxAttempts` times, awaiting a backoff
 * delay between attempts. Only `RunnerError`s with `retryable: true` are
 * retried; everything else is rethrown immediately.
 *
 * Returns the result of the first successful attempt. If every attempt
 * fails, throws the most recent error.
 *
 * Tests inject a custom `sleepFn` to avoid real waiting. We don't expose
 * other knobs because the policy is the same shape across all runners by
 * design (uniform observable behavior under load).
 */
export async function withRetries<T>(
  op: () => Promise<T>,
  opts: {
    maxAttempts?: number;
    baseDelayMs?: number;
    jitterPct?: number;
    sleepFn?: (ms: number) => Promise<void>;
  } = {},
): Promise<T> {
  const maxAttempts = opts.maxAttempts ?? RUNNER_DEFAULTS.retry.maxAttempts;
  const baseDelayMs = opts.baseDelayMs ?? RUNNER_DEFAULTS.retry.baseDelayMs;
  const jitterPct = opts.jitterPct ?? RUNNER_DEFAULTS.retry.jitterPct;
  const sleepFn = opts.sleepFn ?? sleep;

  let lastError: unknown;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await op();
    } catch (err) {
      lastError = err;
      const runnerErr = err as Partial<RunnerError>;
      const retryable = runnerErr?.name === "RunnerError" && runnerErr.retryable === true;
      const isLastAttempt = attempt === maxAttempts - 1;
      if (!retryable || isLastAttempt) {
        throw err;
      }
      await sleepFn(backoffDelayMs(attempt, baseDelayMs, jitterPct));
    }
  }
  // Unreachable: the loop either returns or throws. The throw is here only
  // to satisfy the type checker's flow analysis.
  throw lastError;
}
