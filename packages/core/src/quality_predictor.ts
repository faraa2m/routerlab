// quality_predictor.ts — calibrated quality predictor for routerlab.
//
// --------------------------------------------------------------------------
// WHY THIS MODULE EXISTS
// --------------------------------------------------------------------------
// Phase 2 shipped a hardcoded per-(taskClass, model) quality prior in
// `quality_prior.ts`. That prior was a deliberate placeholder: routing
// decisions were "shaped right" but uncalibrated. Phase 3 replaces it with
// a real, data-driven predictor whose mean and 95% confidence interval are
// computed from the eval-harness measurements written to
// `eval/results/quality_table.json` by the `router-frontier` agent.
//
// Differentiation versus prior art:
// a calibrated pre-call quality estimator is not novel on its own —
// RouteLLM's matrix factorization, BEST-Route's difficulty heads, and
// cross-attention routers all do this. Our differentiation is reporting
// **explicit confidence intervals** (Wilson score, n-aware) so that a
// caller can route on a *lower-bound* quality estimate at high confidence
// rather than a point estimate, and pair that with atlas-grounded cost
// (see `cost.ts`) for a fully accountable routing trace.
//
// --------------------------------------------------------------------------
// DATA SOURCE PRECEDENCE
// --------------------------------------------------------------------------
//   1. `eval/results/quality_table.json` on disk if present (read once,
//      memoized). Each cell carries `{ trials, successes }` from the eval
//      harness. Mean is `successes/trials`; CI is Wilson score 95%.
//      `n` is `trials`.
//   2. Otherwise the seeded prior table is used. The prior is
//      treated as if it were a measurement with `PRIOR_N` synthetic
//      trials so that the CI still has a defined shape. Callers can
//      detect this from the `n` field — it equals `PRIOR_N` when the
//      data is the fallback rather than a real measurement.
//
// --------------------------------------------------------------------------
// PURITY & DETERMINISM
// --------------------------------------------------------------------------
// All math is pure and deterministic. The only side effect is the one-time
// disk read at module init, memoized in `qualityTableCache`. Tests can
// reset that cache via `__resetQualityCacheForTest`.

import { readFileSync } from "node:fs";
import { dirname, resolve as resolvePath } from "node:path";
import { fileURLToPath } from "node:url";
import { __QUALITY_PRIOR_TABLE } from "./quality_prior.ts";
import type { TaskClass } from "./types.ts";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/**
 * A point estimate plus its 95% Wilson-score confidence interval.
 *
 * - `mean`: best-estimate success probability in [0, 1].
 * - `lo95` / `hi95`: 95% Wilson-score interval bounds in [0, 1].
 * - `n`: number of evaluation trials backing the estimate. When the
 *   measurement file is missing this equals `PRIOR_N` (see below) so
 *   callers can detect the fallback case.
 */
export interface QualityWithCI {
  mean: number;
  lo95: number;
  hi95: number;
  n: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Z-score for a 95% confidence interval. Two-sided, normal approximation.
 * Hardcoded to avoid pulling in a stats dependency; the Wilson interval
 * uses this value verbatim.
 */
const Z_95 = 1.959963984540054;

/**
 * Default quality when neither the measured table nor the prior carries
 * an entry for (taskClass, model). Matches `quality_prior.ts`'s default
 * so callers see consistent behaviour across the two paths.
 */
const DEFAULT_QUALITY = 0.5;

/**
 * Synthetic trial count assigned to a fallback-prior cell so that the CI
 * is well-defined. Chosen at 10 trials, which yields a wide CI (~+/-0.3
 * for p=0.5) — that's correct behaviour: the prior should be treated as
 * weak evidence, and a caller routing on `lo95` will be appropriately
 * conservative until real measurements land.
 */
export const PRIOR_N = 10;

/**
 * The default location of the eval-harness quality table on disk.
 * Resolved repo-relative from this module's URL so the predictor works on
 * any machine without hardcoded absolute paths. This file lives at
 * `packages/core/src/quality_predictor.ts`; the quality table is at
 * `eval/results/quality_table.json` (4 levels up to repo root, then down).
 * Tests override via the env var below.
 */
const DEFAULT_QUALITY_TABLE_PATH = resolvePath(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "..",
  "..",
  "..",
  "eval",
  "results",
  "quality_table.json",
);

/**
 * Env var that overrides `DEFAULT_QUALITY_TABLE_PATH`. Tests set this to
 * a tmp file; production callers can use it to ship custom calibrations.
 */
const QUALITY_TABLE_PATH_ENV_VAR = "ROUTERLAB_QUALITY_TABLE_PATH";

// ---------------------------------------------------------------------------
// Wilson score interval (pure math)
// ---------------------------------------------------------------------------

/**
 * Wilson score 95% confidence interval for a binomial proportion.
 *
 * Reference: Wilson, E. B. (1927). "Probable inference, the law of
 * succession, and statistical inference." JASA 22(158): 209-212.
 *
 * Why Wilson over Wald: Wald (the textbook normal approximation, p̂ ±
 * z*sqrt(p̂(1-p̂)/n)) collapses to the empty interval when p̂ ∈ {0, 1}
 * and underestimates uncertainty for small n. Wilson is well-defined at
 * the boundary, has better coverage at small n, and is the recommended
 * default for binomial CIs in modern stats texts (e.g. Agresti & Coull,
 * 1998 — "Approximate is better than 'exact' for interval estimation of
 * binomial proportions").
 *
 * Inputs:
 *   - `successes`: integer in [0, trials].
 *   - `trials`: positive integer.
 *
 * Returns the (lo, hi) tuple clamped to [0, 1]. Pure function.
 */
export function wilsonScore95(successes: number, trials: number): { lo: number; hi: number } {
  if (!Number.isFinite(successes) || !Number.isFinite(trials)) {
    throw new Error("wilsonScore95: successes and trials must be finite numbers");
  }
  if (trials <= 0) {
    throw new Error("wilsonScore95: trials must be a positive integer");
  }
  if (successes < 0 || successes > trials) {
    throw new Error("wilsonScore95: successes must be in [0, trials]");
  }

  const n = trials;
  const p = successes / n;
  const z = Z_95;
  const z2 = z * z;

  const denom = 1 + z2 / n;
  const center = (p + z2 / (2 * n)) / denom;
  const radius = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom;

  // Boundary fix-ups. At p=1 or p=0 the analytical Wilson bounds touch
  // the unit interval exactly, but floating-point evaluation drifts by a
  // few ulps. Snap those cases so callers get exact 0/1 values and so
  // that downstream comparisons (e.g. `hi >= mean` when mean = 1) hold.
  let lo = Math.max(0, center - radius);
  let hi = Math.min(1, center + radius);
  if (successes === trials) hi = 1;
  if (successes === 0) lo = 0;
  return { lo, hi };
}

// ---------------------------------------------------------------------------
// On-disk quality table
// ---------------------------------------------------------------------------

/**
 * One cell of the measured-quality table: a (successes, trials) pair for
 * a (taskClass, model) bucket. The Wilson CI is recomputed at lookup
 * time so the on-disk file stays compact and auditable.
 */
interface QualityCell {
  successes: number;
  trials: number;
}

/**
 * Parsed shape of `eval/results/quality_table.json`.
 *
 * Schema:
 * {
 *   "schema_version": 1,
 *   "generated_at": "<iso8601>",
 *   "cells": {
 *     "<modelId>": {
 *       "<taskClass>": { "successes": <int>, "trials": <int> }
 *     }
 *   }
 * }
 *
 * The schema is intentionally minimal — Wilson CI computation lives in
 * this module, not in the eval harness, so the harness only has to emit
 * raw counts. That keeps the data layer auditable and tooling-friendly.
 */
interface QualityTable {
  source: "measured" | "prior";
  cells: Record<string, Partial<Record<TaskClass, QualityCell>>>;
  loadedFrom?: string;
  generatedAt?: string;
}

interface RawQualityFile {
  schema_version?: number;
  generated_at?: string;
  cells?: Record<
    string,
    Record<string, { successes?: unknown; trials?: unknown }>
  >;
}

const TASK_CLASSES: ReadonlySet<TaskClass> = new Set<TaskClass>([
  "qa",
  "codegen",
  "summarization",
  "classification",
  "reasoning",
]);

const isTaskClass = (s: string): s is TaskClass => TASK_CLASSES.has(s as TaskClass);

const parseQualityFile = (path: string, raw: string): QualityTable => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (cause) {
    throw new Error(
      `quality_table.json at "${path}" is not valid JSON: ${
        cause instanceof Error ? cause.message : String(cause)
      }`
    );
  }
  if (parsed === null || typeof parsed !== "object") {
    throw new Error(`quality_table.json at "${path}" root must be an object`);
  }
  const file = parsed as RawQualityFile;
  const cellsIn = file.cells;
  if (cellsIn === undefined || typeof cellsIn !== "object" || cellsIn === null) {
    // Schema-compatible but empty: treat as no measurements present and
    // fall back to the prior, mirroring cost.ts's tolerance of a partly
    // populated atlas file. The caller will see `source: "prior"` cells.
    return buildPriorTable();
  }

  const cellsOut: Record<string, Partial<Record<TaskClass, QualityCell>>> = {};
  for (const [modelId, perTask] of Object.entries(cellsIn)) {
    if (typeof perTask !== "object" || perTask === null) continue;
    const inner: Partial<Record<TaskClass, QualityCell>> = {};
    for (const [taskKey, cell] of Object.entries(perTask)) {
      if (!isTaskClass(taskKey)) continue;
      if (typeof cell !== "object" || cell === null) continue;
      const successes = (cell as { successes?: unknown }).successes;
      const trials = (cell as { trials?: unknown }).trials;
      if (
        typeof successes !== "number" ||
        !Number.isFinite(successes) ||
        successes < 0
      ) {
        continue;
      }
      if (typeof trials !== "number" || !Number.isFinite(trials) || trials <= 0) {
        continue;
      }
      if (successes > trials) continue;
      inner[taskKey] = {
        successes: Math.floor(successes),
        trials: Math.floor(trials),
      };
    }
    if (Object.keys(inner).length > 0) {
      cellsOut[modelId] = inner;
    }
  }

  if (Object.keys(cellsOut).length === 0) {
    // File parsed but produced no usable cells — same fallback as missing.
    return buildPriorTable();
  }

  return {
    source: "measured",
    cells: cellsOut,
    loadedFrom: path,
    ...(typeof file.generated_at === "string" ? { generatedAt: file.generated_at } : {}),
  };
};

/**
 * Build a `QualityTable` from the hardcoded prior. The prior values
 * are interpreted as probabilities; we synthesize `(successes, trials)`
 * with `trials = PRIOR_N` so the Wilson CI has a defined shape.
 *
 * The synthesized successes are clamped to `[1, PRIOR_N - 1]`. Intuition:
 * the prior is **weak evidence**, never a certainty. A cell that says
 * "0.95 quality" should not back the implausibly strong claim "perfect
 * on 10/10 synthetic trials" — that would convince a strict caller
 * (`qualityBar = 1.0`) to route to the model with no real measurements
 * to support it. Capping at `PRIOR_N - 1` keeps the prior conservative
 * and ensures a `qualityBar = 1.0` request always falls through to real
 * data or errors out — both of which are the correct behaviour.
 */
const buildPriorTable = (): QualityTable => {
  const cells: Record<string, Partial<Record<TaskClass, QualityCell>>> = {};
  for (const [modelId, row] of Object.entries(__QUALITY_PRIOR_TABLE)) {
    const inner: Partial<Record<TaskClass, QualityCell>> = {};
    for (const taskKey of TASK_CLASSES) {
      const p = row[taskKey];
      // Round half-up, then clamp to the open interval (0, PRIOR_N).
      const rounded = Math.round(p * PRIOR_N);
      const successes = Math.min(PRIOR_N - 1, Math.max(1, rounded));
      inner[taskKey] = { successes, trials: PRIOR_N };
    }
    cells[modelId] = inner;
  }
  return { source: "prior", cells };
};

// ---------------------------------------------------------------------------
// Memoization
// ---------------------------------------------------------------------------

let qualityTableCache: QualityTable | undefined;

const resolveQualityTablePath = (): string => {
  const fromEnv = process.env[QUALITY_TABLE_PATH_ENV_VAR];
  if (fromEnv !== undefined && fromEnv.length > 0) return fromEnv;
  return DEFAULT_QUALITY_TABLE_PATH;
};

const loadQualityTable = (): QualityTable => {
  const path = resolveQualityTablePath();
  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch {
    // File not present — the expected state before the eval harness runs.
    // Fall back to the seeded prior so routing stays usable.
    return buildPriorTable();
  }
  return parseQualityFile(path, raw);
};

const getQualityTable = (): QualityTable => {
  if (qualityTableCache !== undefined) return qualityTableCache;
  qualityTableCache = loadQualityTable();
  return qualityTableCache;
};

/**
 * Test-only hook: clears the memoized quality table so the next lookup
 * re-reads disk. Mirrors `__resetCalibrationCacheForTest` in `cost.ts`.
 * Not part of the public API surface but intentionally exported with the
 * `__`-prefix convention so tests can reach for it explicitly.
 */
export const __resetQualityCacheForTest = (): void => {
  qualityTableCache = undefined;
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Return the expected quality for `(taskClass, modelId)` as a point estimate
 * in [0, 1].
 *
 * Backward-compatible signature: matches `quality_prior.predictQuality` so
 * the router doesn't need to change. The mean comes from
 * `predictQualityWithCI` — callers wanting uncertainty should use that
 * function directly.
 *
 * Unknown models receive `DEFAULT_QUALITY = 0.5`.
 */
export function predictQuality(taskClass: TaskClass, modelId: string): number {
  return predictQualityWithCI(taskClass, modelId).mean;
}

/**
 * Return the expected quality for `(taskClass, modelId)` with a 95%
 * Wilson-score confidence interval.
 *
 * - `mean` is `successes / trials`.
 * - `lo95` / `hi95` is the Wilson score interval at the 95% level.
 * - `n` is the trial count (real for measured cells, `PRIOR_N` for the
 *   seeded prior fallback).
 *
 * Unknown (taskClass, modelId) pairs return a uniform-prior estimate:
 *   `{ mean: 0.5, lo95, hi95, n: PRIOR_N }`
 * where the CI is the Wilson 95% interval for 5 successes in 10 trials —
 * intentionally wide to reflect that no data backs the estimate.
 */
export function predictQualityWithCI(
  taskClass: TaskClass,
  modelId: string
): QualityWithCI {
  const table = getQualityTable();
  const row = table.cells[modelId];
  const cell = row !== undefined ? row[taskClass] : undefined;
  if (cell === undefined) {
    // Unknown model or unknown task class for a known model: return a
    // uniform prior with PRIOR_N synthetic trials so the CI is defined.
    const successes = Math.round(DEFAULT_QUALITY * PRIOR_N);
    const { lo, hi } = wilsonScore95(successes, PRIOR_N);
    return {
      mean: DEFAULT_QUALITY,
      lo95: lo,
      hi95: hi,
      n: PRIOR_N,
    };
  }
  const mean = cell.successes / cell.trials;
  const { lo, hi } = wilsonScore95(cell.successes, cell.trials);
  return { mean, lo95: lo, hi95: hi, n: cell.trials };
}

/**
 * Introspection helper. Returns whether the predictor is currently serving
 * measurements or the fallback prior, and the metadata of the loaded file.
 * Useful for the CLI's `route --debug` output and for the paper's
 * reproducibility appendix.
 */
export function getQualitySourceInfo(): {
  source: "measured" | "prior";
  loadedFrom?: string;
  generatedAt?: string;
} {
  const table = getQualityTable();
  const info: { source: "measured" | "prior"; loadedFrom?: string; generatedAt?: string } = {
    source: table.source,
  };
  if (table.loadedFrom !== undefined) info.loadedFrom = table.loadedFrom;
  if (table.generatedAt !== undefined) info.generatedAt = table.generatedAt;
  return info;
}
