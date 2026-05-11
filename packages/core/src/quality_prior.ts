// quality_prior.ts — seeded fallback prior for the routing engine.
//
// PHASE 2 -> PHASE 3 ROLE SHIFT: This file used to be the source of truth
// for `predictQuality`. In Phase 3 it has been demoted to a fallback prior
// consumed by `quality_predictor.ts`. The real, calibrated quality
// estimates now come from measured eval data at
// `eval/results/quality_table.json`. When that file is missing or
// produces no usable cells (e.g. project bootstrap, fresh checkout
// before the eval harness has run), `quality_predictor.ts` synthesizes
// (successes, trials) counts from this prior table so the Wilson CI is
// well-defined. The `predictQuality()` function below is kept for
// backward-compat — `index.ts` now re-exports the predictor's version
// in preference — but the table data is the load-bearing artifact.
//
// Prior-art note: a calibrated
// pre-call quality predictor on its own is not novel — RouteLLM's matrix
// factorization, BEST-Route's difficulty heads, and cross-attention
// routers all do this. Our differentiation comes from pairing the
// predictor with atlas-grounded empirical token costs PLUS reporting
// explicit Wilson 95% CIs (so callers can route on a confident lower
// bound rather than a point estimate), NOT from the predictor's
// algorithmic structure.
//
// Design choices for the prior values:
//   - Bigger / newer models score higher on harder tasks (codegen,
//     reasoning) where parameter count and training recency dominate.
//   - Smaller / cheaper models score competitively on easier tasks
//     (classification, summarization, simple QA) where capability
//     headroom is wasted.
//   - Values are eyeballed from public eval reports (MT-Bench, HumanEval,
//     MMLU, etc.) circa late-2025 / early-2026 — they are NOT measured
//     here. Phase 3 replaces them with measurements.

import type { TaskClass } from "./types.ts";

/**
 * Internal type: a quality estimate per task class for a single model id.
 * Missing entries fall back to `DEFAULT_QUALITY`.
 */
type QualityRow = Record<TaskClass, number>;

const DEFAULT_QUALITY = 0.5;

/**
 * Hardcoded per-(model, task) quality prior. The key is the model id from
 * the candidate pool (see `candidates.json`). Values are expected quality
 * in [0, 1] interpreted as "fraction of tasks of this class that this
 * model gets right at the rubric's threshold."
 *
 * REPLACE LATER with a calibrated predictor; see the prior-art
 * survey for differentiation requirements.
 */
const QUALITY_PRIOR: Record<string, QualityRow> = {
  "claude-opus-4-7": {
    qa: 0.95,
    codegen: 0.93,
    summarization: 0.94,
    classification: 0.96,
    reasoning: 0.95,
  },
  "claude-sonnet-4-6": {
    qa: 0.91,
    codegen: 0.88,
    summarization: 0.92,
    classification: 0.93,
    reasoning: 0.89,
  },
  "claude-haiku-4-5": {
    qa: 0.84,
    codegen: 0.78,
    summarization: 0.86,
    classification: 0.89,
    reasoning: 0.78,
  },
  "llama-3.3-70b": {
    qa: 0.82,
    codegen: 0.79,
    summarization: 0.83,
    classification: 0.87,
    reasoning: 0.76,
  },
  "llama-3.1-8b": {
    qa: 0.68,
    codegen: 0.58,
    summarization: 0.72,
    classification: 0.78,
    reasoning: 0.55,
  },
  "mixtral-8x7b": {
    qa: 0.74,
    codegen: 0.69,
    summarization: 0.77,
    classification: 0.81,
    reasoning: 0.65,
  },
};

/**
 * Look up the expected quality for `(taskClass, modelId)` from the seeded
 * prior table. Kept exported for backward-compat with older callers and
 * for tests that need to inspect the raw prior. New code should prefer
 * `predictQuality` / `predictQualityWithCI` from `quality_predictor.ts`,
 * which serves measured data when available and falls back to this
 * table otherwise.
 *
 * Models not in the table fall back to `DEFAULT_QUALITY = 0.5`.
 */
export function predictQuality(taskClass: TaskClass, modelId: string): number {
  const row = QUALITY_PRIOR[modelId];
  if (row === undefined) {
    return DEFAULT_QUALITY;
  }
  return row[taskClass];
}

/**
 * Exposed for tests and for introspection by downstream tooling (e.g. the
 * Phase 3 calibration harness, which uses this as a starting prior).
 */
export const __QUALITY_PRIOR_TABLE: Readonly<Record<string, QualityRow>> =
  QUALITY_PRIOR;
