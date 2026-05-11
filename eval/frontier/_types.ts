// eval/frontier/_types.ts — shared types for the frontier pipeline.
//
// The frontier pipeline produces routerlab's load-bearing public artifact:
// `eval/results/frontier.json`. This is the file no prior open router
// publishes — per-task-class Pareto frontiers with **atlas-grounded cost
// attribution** baked in.
//
// The pipeline has three stages, each with its own type:
//
//   1. `RunOutcome` — one (task, model, example) tuple's measurement.
//      Persisted to `eval/results/runs/{task}/{model}/{example_id}.json`.
//   2. `SummaryRow` — per-(task, model) aggregate. Persisted to
//      `eval/results/summary.json` as an array of these.
//   3. `FrontierEntry` / `FrontierFile` — the Pareto frontier per task
//      class, derived from `summary.json`. This is the published artifact.
//
// All types are strict TS, no `any`, no opaque `unknown` at boundaries.
// Stable string literal unions for task classes and provider ids are
// re-exported here so consumers of the frontier file don't have to import
// from `packages/core/src/types.ts` or `cost.ts`.

import type { TaskClass } from "../../packages/core/src/types.ts";
import type { Confidence, TokenSource } from "../../packages/core/src/cost.ts";

// Re-export TaskClass + cost-attribution enums for downstream consumers
// (e.g. the paper's analysis scripts, the quality predictor) so they can
// type their frontier-file readers against a single source of truth.
export type { TaskClass, TokenSource, Confidence };

/**
 * A persisted measurement of running one example on one model. Captures
 * everything needed for an audit trail: the prompt hash, the raw output,
 * the parsed output, the score, the cost (with provenance), latency, and
 * the source of the quality score (rubric vs LLM judge).
 *
 * Storage layout:
 *   `eval/results/runs/{task}/{model}/{example_id}.json`
 *
 * The filename uses the example_id verbatim so re-running a sweep with
 * the same seed is idempotent (overwrites the same file).
 */
export interface RunOutcome {
  /** Stable task class id (matches `TaskClass` in `@routerlab/core`). */
  task: TaskClass;
  /** Canonical model id (matches `candidates.json`). */
  model: string;
  /** Provider hosting the model (anthropic, groq, together, hf, openrouter). */
  provider: string;
  /** Example id from the task's `loadExamples()` pool. */
  exampleId: string;
  /**
   * SHA-256-truncated hex hash of the rendered prompt. Used to verify the
   * same prompt was scored across re-runs even if `loadExamples()` reshuffles.
   */
  promptHash: string;
  /** The model's raw text output. */
  output: string;
  /** Score in [0, 1] from the scoring path used (see `scoreSource`). */
  qualityScore: number;
  /** How the quality score was produced. */
  scoreSource: "task-score" | "judge";
  /** USD cost for this single call. */
  costUsd: number;
  /** Token-source attribution from `estimateCost`. Audit-grade. */
  tokenSource: TokenSource;
  /** Confidence label from `estimateCost`. */
  confidence: Confidence;
  /** Provider-reported input tokens, if available. */
  inputTokens?: number;
  /** Provider-reported output tokens, if available. */
  outputTokens?: number;
  /** Wall-clock latency in ms. */
  latencyMs: number;
  /** ISO-8601 timestamp at run completion. */
  ts: string;
  /** Set when the run errored — the rest of the fields are best-effort. */
  error?: string;
}

/**
 * Per-(task, model) aggregate. Computed by the build orchestrator after the
 * raw outcomes for a sweep have been persisted.
 *
 * Persisted to `eval/results/summary.json` as a flat array:
 *   `[ { task: "qa", model: "...", ... }, ... ]`
 *
 * The flat-array shape is deliberately denormalized — keeps the file easy
 * to load + filter in any language (jq, pandas, polars) without having to
 * traverse a nested map.
 */
export interface SummaryRow {
  task: TaskClass;
  model: string;
  provider: string;
  /** Number of successful (non-errored) measurements aggregated here. */
  n: number;
  /** Mean quality score in [0, 1]. */
  mean_quality: number;
  /** Mean USD cost per call. */
  mean_cost_usd: number;
  /** Median quality score in [0, 1]. */
  p50_quality: number;
  /** Median USD cost per call. */
  p50_cost_usd: number;
  /** 95th percentile quality score. */
  p95_quality: number;
  /** 95th percentile USD cost per call. */
  p95_cost_usd: number;
  /**
   * Token-source attribution. If all per-call estimates agreed, this is
   * the agreed value; otherwise the most-common value with a note in
   * `tokenSourceNote`. Audit-grade.
   */
  tokenSource: TokenSource;
  /** Confidence label, same aggregation rule as `tokenSource`. */
  confidence: Confidence;
  /** Set when per-call attributions disagreed; nil otherwise. */
  tokenSourceNote?: string;
  /** Number of errored calls excluded from `n` and the aggregates. */
  errors: number;
}

/**
 * One entry in the per-task Pareto frontier. Shape is kept minimal so
 * the published `frontier.json` stays readable + auditable. Token-source
 * provenance is retained from the underlying summary row so any reader
 * can verify atlas-grounding.
 */
export interface FrontierEntry {
  model: string;
  provider: string;
  mean_quality: number;
  mean_cost_usd: number;
  n: number;
  tokenSource: TokenSource;
  confidence: Confidence;
}

/**
 * One row in the "all candidates" listing per task. Same shape as a
 * frontier entry plus a `dominated` boolean flagging whether the row is
 * on the frontier or dominated by another model.
 */
export interface AllEntry extends FrontierEntry {
  dominated: boolean;
}

/**
 * Per-task block of the published frontier file. `frontier` is the
 * Pareto-optimal subset sorted by ascending cost; `all` is every model
 * we measured for the task (frontier + dominated), same sort.
 */
export interface FrontierTaskBlock {
  frontier: FrontierEntry[];
  all: AllEntry[];
  /** Models classified as dominated (subset of `all` with `dominated=true`). */
  dominated: AllEntry[];
}

/**
 * Top-level shape of `eval/results/frontier.json` — the load-bearing
 * published artifact. Keyed by task class. Includes meta fields documenting
 * the cost-source so audits can verify the atlas-grounding claim
 * without re-running the eval.
 */
export interface FrontierFile {
  schema_version: 1;
  generated_at: string;
  /**
   * Provenance breadcrumb: confirms cost was computed via `cost.ts`
   * (atlas-grounded path) rather than a chars/4 proxy.
   */
  cost_source: {
    module: "@routerlab/core/cost";
    atlas_results_path?: string;
    /** Aggregate distribution of token-sources across all measured cells. */
    token_source_distribution: Record<TokenSource, number>;
    /** Aggregate distribution of confidence levels across all measured cells. */
    confidence_distribution: Record<Confidence, number>;
  };
  tasks: Record<TaskClass, FrontierTaskBlock>;
}
