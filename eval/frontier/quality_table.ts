// eval/frontier/quality_table.ts — derive `quality_table.json` from `summary.json`.
//
// `packages/core/src/quality_predictor.ts` consumes this file to provide
// the routing engine with measured per-(task, model) quality estimates
// (with 95% Wilson-score CIs). The on-disk schema the predictor reads is:
//
//   {
//     "schema_version": 1,
//     "generated_at": "<iso8601>",
//     "cells": {
//       "<modelId>": {
//         "<taskClass>": { "successes": <int>, "trials": <int> }
//       }
//     }
//   }
//
// The orchestrator persists `summary.json` with per-(task, model) means
// and counts. The brief asks for `{(task, model): {mean, n}}` derivation
// — we satisfy both contracts by emitting `trials = n` and `successes =
// round(mean * n)`. That keeps the predictor's expected shape intact and
// gives the predictor non-trivial counts to feed into the Wilson CI math.
//
// Edge cases handled here:
//   - n === 0 (every measurement errored) → cell is OMITTED so the
//     predictor falls back to its prior table for that bucket. Emitting
//     an n=0 cell would trip the predictor's "trials must be positive"
//     guard.
//   - mean outside [0, 1] (shouldn't happen — `runOne` clamps) → cell is
//     omitted to be defensive.
//   - mean * n produces a non-integer → rounded half-up to the nearest
//     integer, then clamped to [0, n].
//
// Pure-IO module: reads `summary.json`, writes `quality_table.json`. No
// network. No model calls.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { TaskClass } from "../../packages/core/src/types.ts";

import type { SummaryRow } from "./_types.ts";

const DEFAULT_RESULTS_DIR: string = (() => {
  const here = dirname(fileURLToPath(import.meta.url));
  return join(here, "..", "results");
})();

/**
 * One cell of the on-disk quality table.
 */
export interface QualityTableCell {
  successes: number;
  trials: number;
}

/**
 * Persisted shape, matching what the predictor parses.
 */
export interface QualityTableFile {
  schema_version: 1;
  generated_at: string;
  cells: Record<string, Partial<Record<TaskClass, QualityTableCell>>>;
}

// ---------------------------------------------------------------------------
// Pure derivation
// ---------------------------------------------------------------------------

/**
 * Build the in-memory `QualityTableFile` from a flat array of summary rows.
 *
 * The function is pure: same input → same output (modulo `generated_at`,
 * which the caller can override for byte-stable test output).
 */
export function buildQualityTable(
  rows: readonly SummaryRow[],
  opts: { generatedAt?: string } = {},
): QualityTableFile {
  const cells: Record<string, Partial<Record<TaskClass, QualityTableCell>>> = {};

  // Deterministic walk: sort by (task, model) so the on-disk JSON ordering
  // matches across re-runs given the same input. JSON.stringify on plain
  // objects is insertion-ordered in V8 + Bun, so this matters.
  const sortedRows = rows.slice().sort((a, b) => {
    if (a.task !== b.task) return a.task < b.task ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });

  for (const row of sortedRows) {
    if (row.n <= 0) continue;
    if (!Number.isFinite(row.mean_quality)) continue;
    if (row.mean_quality < 0 || row.mean_quality > 1) continue;

    const trials = Math.floor(row.n);
    if (trials <= 0) continue;
    const rawSuccesses = row.mean_quality * trials;
    const successes = Math.min(trials, Math.max(0, Math.round(rawSuccesses)));

    const perModel = cells[row.model] ?? {};
    perModel[row.task] = { successes, trials };
    cells[row.model] = perModel;
  }

  return {
    schema_version: 1,
    generated_at: opts.generatedAt ?? new Date().toISOString(),
    cells,
  };
}

// ---------------------------------------------------------------------------
// IO entrypoint
// ---------------------------------------------------------------------------

function readSummary(path: string): SummaryRow[] {
  if (!existsSync(path)) {
    throw new Error(
      `summary.json not found at "${path}" — run the frontier sweep first`,
    );
  }
  const raw = readFileSync(path, "utf8");
  const parsed = JSON.parse(raw) as { rows?: SummaryRow[] };
  if (parsed.rows === undefined || !Array.isArray(parsed.rows)) {
    throw new Error(
      `summary.json at "${path}" is missing a top-level "rows" array`,
    );
  }
  return parsed.rows;
}

export interface BuildQualityTableFromDiskOptions {
  resultsDir?: string;
  /** Override the generated_at timestamp (tests use a fixed string). */
  generatedAt?: string;
}

/**
 * Read `summary.json`, derive the quality table, and write
 * `quality_table.json`. Returns the in-memory file for tests.
 */
export function buildQualityTableFromDisk(
  opts: BuildQualityTableFromDiskOptions = {},
): QualityTableFile {
  const resultsDir = opts.resultsDir ?? DEFAULT_RESULTS_DIR;
  const summaryPath = join(resultsDir, "summary.json");
  const rows = readSummary(summaryPath);

  const tableOpts: { generatedAt?: string } = {};
  if (opts.generatedAt !== undefined) {
    tableOpts.generatedAt = opts.generatedAt;
  }
  const file = buildQualityTable(rows, tableOpts);

  const outPath = join(resultsDir, "quality_table.json");
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(file, null, 2) + "\n", "utf8");
  return file;
}
