// eval/judge/_types.ts — shared types for the LLM-as-judge harness.
//
// The judge is used by `router-frontier` (Phase 3 sibling) to score model
// outputs on task classes where reference-based metrics (F1, ROUGE-L,
// exact-match) under-measure quality — chiefly codegen and summarization.
// The harness is parametric over task class and rubric so the same judge
// model can be reused across all five task classes.
//
// Contract is intentionally narrow:
//   - `JudgeRequest` carries everything the judge needs to issue a verdict:
//     the candidate output, the original prompt, optionally the gold
//     reference, and an optional task-specific rubric.
//   - `JudgeResponse` carries the normalized 0..1 score, the judge's
//     reasoning, the judge model id, a cache-hit boolean (for cost
//     accounting), and a timestamp.
//
// No `any`. Strict TS only.

import type { TaskClass } from "../../packages/core/src/types.ts";

/**
 * Input to one judge call. `taskClass` selects the default rubric template
 * from `prompts.ts`; callers can override via `rubric` for task-specific
 * tweaks (e.g. "favor concise summaries" for a summarization sweep).
 *
 * `reference` is `unknown` because each task class encodes its gold
 * reference in a different shape (string for QA, label for classification,
 * `CodegenReference` for codegen, etc.). The judge prompt template is
 * responsible for stringifying it sensibly.
 */
export interface JudgeRequest {
  taskClass: TaskClass;
  /** Original task prompt given to the candidate model. */
  prompt: string;
  /** Candidate model's raw output. */
  candidate: string;
  /** Gold reference, if available. Shape depends on `taskClass`. */
  reference?: unknown;
  /** Optional override of the per-task-class default rubric. */
  rubric?: string;
}

/**
 * Output of one judge call. `score` is normalized to [0,1] (the underlying
 * judge model produces a 0-10 raw score which the harness divides by 10
 * with bounds checking).
 *
 * `cacheHit` signals whether this response was served from the on-disk
 * cache. The orchestrator uses this to compute the cost of an eval sweep
 * accurately (cache hits cost nothing).
 */
export interface JudgeResponse {
  /** Normalized score in [0, 1]. */
  score: number;
  /** Brief judge reasoning, useful for debugging odd scores. */
  reasoning: string;
  /** Canonical model id of the judge that produced this response. */
  judge_model: string;
  /** True iff this response was loaded from the on-disk cache. */
  cacheHit: boolean;
  /** ISO-8601 timestamp at the moment of response (cache write or live call). */
  ts: string;
}

/**
 * Options bag for `judge()`. The judge model defaults to `claude-haiku-4-5`
 * (the cheapest competent option per the project plan).
 *
 * `cacheDir` defaults to `eval/judge/cache/`. Tests inject a tmp dir so
 * they don't pollute the real cache (and the real cache doesn't pollute
 * test runs).
 *
 * `client` is the same kind of injectable SDK stub the anthropic runner
 * accepts — set it in tests to avoid hitting the network. In production
 * leave it undefined and the harness builds its own SDK client from
 * `ANTHROPIC_API_KEY`.
 */
export interface JudgeOptions {
  /** Override the judge model. Default: `claude-haiku-4-5`. */
  judgeModel?: string;
  /** Override the cache directory. Default: `eval/judge/cache/`. */
  cacheDir?: string;
  /** Disable the cache entirely (force a live call). Default: false. */
  noCache?: boolean;
  /** Override the Anthropic API key (else read from env). */
  apiKey?: string;
  /**
   * Inject a stub anthropic SDK client for tests. Same shape as the one
   * accepted by the anthropic runner; the harness uses only
   * `messages.create`.
   */
  client?: {
    messages: {
      create: (...args: unknown[]) => Promise<unknown>;
    };
  };
  /** Override the per-call timeout. */
  timeoutMs?: number;
  /** Inject a custom retry policy (tests pass `sleepFn` to skip waits). */
  retry?: {
    maxAttempts?: number;
    baseDelayMs?: number;
    jitterPct?: number;
    sleepFn?: (ms: number) => Promise<void>;
  };
}

/** Default judge model — cheapest competent Anthropic tier in 2026. */
export const DEFAULT_JUDGE_MODEL = "claude-haiku-4-5" as const;

/** Default cache directory relative to the repo root. */
export const DEFAULT_CACHE_DIR_RELATIVE = "eval/judge/cache" as const;
