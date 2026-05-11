// eval/frontier/build_frontier.ts — Pareto frontier construction.
//
// Given `eval/results/summary.json` (an array of `SummaryRow`), compute
// the per-task-class Pareto frontier and emit `eval/results/frontier.json`.
//
// The frontier is the load-bearing artifact for routerlab's novelty claim.
// What makes it citable:
//
//   1. The cost axis is atlas-grounded — every row carries the
//      `tokenSource` and `confidence` attribution from `cost.ts`, and
//      the aggregate distribution of those attributions is recorded in
//      the file's `cost_source` block.
//
//   2. The frontier is per **task class** — no other open router publishes
//      one Pareto curve per task. RouterArena, RouteLLM, RouterBench all
//      publish either aggregated routing-decision metrics or single-task
//      leaderboards.
//
//   3. The construction is **deterministic** — given a fixed
//      `summary.json`, the output `frontier.json` is byte-identical
//      across runs (modulo `generated_at`).
//
// Algorithm:
//   - Group summary rows by task class.
//   - Within each group:
//     a. Stable-sort by ascending `mean_cost_usd`, then by descending
//        `mean_quality` (tie-break), then by model id (lexicographic
//        tie-break for total determinism).
//     b. Walk the sorted list. A row is on the frontier iff its
//        `mean_quality` strictly exceeds the highest `mean_quality` seen
//        in any cheaper-or-equal-cost row already on the frontier.
//        (Equivalently: it is not dominated by any other row.)
//   - Emit one block per task with `frontier`, `all` (with `dominated`
//     flag), and `dominated`.
//
// Edge cases:
//   - Task class with zero rows in the summary → the block is emitted
//     with empty arrays, not omitted. Stable schema.
//   - Rows with `n=0` (every measurement errored) are treated as
//     uninformative and excluded from the frontier candidate pool but
//     still listed in `all` with `dominated=true` so the absence isn't
//     silent. They're flagged with `dominated=true` regardless of
//     position.
//   - Rows with identical (cost, quality) keep both in `all`; only one
//     can be on the frontier (the one with the lexicographically smaller
//     model id wins, for determinism).

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { TaskClass } from "../../packages/core/src/types.ts";

import type {
  AllEntry,
  Confidence,
  FrontierEntry,
  FrontierFile,
  FrontierTaskBlock,
  SummaryRow,
  TokenSource,
} from "./_types.ts";

const ALL_TASK_CLASSES: readonly TaskClass[] = [
  "qa",
  "classification",
  "codegen",
  "summarization",
  "reasoning",
];

const ALL_TOKEN_SOURCES: readonly TokenSource[] = [
  "tokenometer-empirical",
  "tokenometer-offline",
  "atlas-calibrated",
  "proxy",
];

const ALL_CONFIDENCES: readonly Confidence[] = ["high", "medium", "low"];

// ---------------------------------------------------------------------------
// Default paths
// ---------------------------------------------------------------------------

const DEFAULT_RESULTS_DIR: string = (() => {
  const here = dirname(fileURLToPath(import.meta.url));
  return join(here, "..", "results");
})();

// ---------------------------------------------------------------------------
// Pure frontier computation
// ---------------------------------------------------------------------------

/**
 * Compute the Pareto frontier and the dominated set for one task class
 * given a list of summary rows. Pure: no IO, deterministic, total
 * function on its inputs.
 *
 * The "dominated" relation: row A dominates row B iff
 *   A.mean_cost_usd <= B.mean_cost_usd AND A.mean_quality > B.mean_quality
 * OR
 *   A.mean_cost_usd < B.mean_cost_usd AND A.mean_quality >= B.mean_quality
 * (strict in at least one dimension). A row is on the frontier iff no
 * other row dominates it.
 *
 * We compute this in O(n log n) by sweeping cost-ascending and tracking
 * the running max quality. A row joins the frontier iff its quality
 * strictly exceeds the running max. Ties at the same (cost, quality)
 * are broken by model id (lexicographic) — only the smallest-id wins
 * the frontier slot.
 */
export function computeParetoFrontier(rows: readonly SummaryRow[]): {
  frontier: FrontierEntry[];
  all: AllEntry[];
  dominated: AllEntry[];
} {
  // Filter out informative rows (n > 0) for the dominance computation,
  // but keep `n === 0` rows in `all` flagged as dominated so the absence
  // of measurement isn't silent.
  const informative = rows.filter((r) => r.n > 0);
  const uninformative = rows.filter((r) => r.n === 0);

  // Stable, total-order sort key for determinism:
  //   1. ascending mean_cost_usd
  //   2. descending mean_quality (so the better-quality row at equal cost
  //      wins the frontier slot)
  //   3. ascending model id (final tie-break for byte-stable output)
  const sorted = informative.slice().sort((a, b) => {
    if (a.mean_cost_usd !== b.mean_cost_usd) {
      return a.mean_cost_usd - b.mean_cost_usd;
    }
    if (a.mean_quality !== b.mean_quality) {
      return b.mean_quality - a.mean_quality;
    }
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });

  const frontierSet = new Set<string>();
  let runningMaxQuality = Number.NEGATIVE_INFINITY;
  for (const row of sorted) {
    if (row.mean_quality > runningMaxQuality) {
      frontierSet.add(`${row.model}|${row.provider}`);
      runningMaxQuality = row.mean_quality;
    }
  }

  const toFrontierEntry = (r: SummaryRow): FrontierEntry => ({
    model: r.model,
    provider: r.provider,
    mean_quality: r.mean_quality,
    mean_cost_usd: r.mean_cost_usd,
    n: r.n,
    tokenSource: r.tokenSource,
    confidence: r.confidence,
  });

  const frontier: FrontierEntry[] = [];
  const all: AllEntry[] = [];

  for (const r of sorted) {
    const key = `${r.model}|${r.provider}`;
    const onFrontier = frontierSet.has(key);
    const entry = toFrontierEntry(r);
    all.push({ ...entry, dominated: !onFrontier });
    if (onFrontier) {
      frontier.push(entry);
    }
  }
  // Append uninformative (n === 0) rows to `all` at the end, sorted by
  // model id for determinism. These are not eligible for the frontier.
  const uninformativeSorted = uninformative.slice().sort((a, b) => {
    if (a.mean_cost_usd !== b.mean_cost_usd) {
      return a.mean_cost_usd - b.mean_cost_usd;
    }
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
  for (const r of uninformativeSorted) {
    all.push({ ...toFrontierEntry(r), dominated: true });
  }

  const dominated = all.filter((a) => a.dominated);
  return { frontier, all, dominated };
}

// ---------------------------------------------------------------------------
// Top-level builder
// ---------------------------------------------------------------------------

/**
 * Compute aggregate provenance distributions over a set of summary rows.
 * Used for the `cost_source` block in the frontier file so downstream
 * auditors can see at a glance what fraction of cells were atlas-
 * calibrated vs offline-only vs proxy.
 */
function aggregateProvenance(rows: readonly SummaryRow[]): {
  token_source_distribution: Record<TokenSource, number>;
  confidence_distribution: Record<Confidence, number>;
} {
  const tokenInit = {} as Record<TokenSource, number>;
  for (const k of ALL_TOKEN_SOURCES) tokenInit[k] = 0;
  const confInit = {} as Record<Confidence, number>;
  for (const k of ALL_CONFIDENCES) confInit[k] = 0;
  for (const r of rows) {
    tokenInit[r.tokenSource] += 1;
    confInit[r.confidence] += 1;
  }
  return {
    token_source_distribution: tokenInit,
    confidence_distribution: confInit,
  };
}

/**
 * Build the frontier file in memory from a flat array of summary rows.
 * Pure function — pulled out for testability.
 */
export function buildFrontierFile(
  rows: readonly SummaryRow[],
  opts: { generatedAt?: string; atlasResultsPath?: string } = {},
): FrontierFile {
  const tasks = {} as Record<TaskClass, FrontierTaskBlock>;
  for (const taskClass of ALL_TASK_CLASSES) {
    const taskRows = rows.filter((r) => r.task === taskClass);
    tasks[taskClass] = computeParetoFrontier(taskRows);
  }

  const provenance = aggregateProvenance(rows);

  const file: FrontierFile = {
    schema_version: 1,
    generated_at: opts.generatedAt ?? new Date().toISOString(),
    cost_source: {
      module: "@routerlab/core/cost",
      ...(opts.atlasResultsPath !== undefined
        ? { atlas_results_path: opts.atlasResultsPath }
        : {}),
      token_source_distribution: provenance.token_source_distribution,
      confidence_distribution: provenance.confidence_distribution,
    },
    tasks,
  };
  return file;
}

// ---------------------------------------------------------------------------
// IO entrypoint
// ---------------------------------------------------------------------------

function readSummaryFile(path: string): SummaryRow[] {
  if (!existsSync(path)) {
    throw new Error(
      `summary.json not found at "${path}" — run the frontier sweep first (bun eval/cli.ts frontier-all)`,
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

export interface BuildFrontierFromDiskOptions {
  /** Override the results dir. Default: `eval/results`. */
  resultsDir?: string;
  /** Override the atlas results path recorded in the file. */
  atlasResultsPath?: string;
  /** Override the generated_at timestamp (tests use a fixed string). */
  generatedAt?: string;
}

/**
 * Read `summary.json`, compute the frontier, and write `frontier.json`.
 * Returns the in-memory `FrontierFile` for convenience (tests assert on
 * it directly).
 */
export function buildFrontierFromDisk(
  opts: BuildFrontierFromDiskOptions = {},
): FrontierFile {
  const resultsDir = opts.resultsDir ?? DEFAULT_RESULTS_DIR;
  const summaryPath = join(resultsDir, "summary.json");
  const rows = readSummaryFile(summaryPath);

  const atlasPathFromEnv = process.env["ROUTERLAB_ATLAS_RESULTS_PATH"];
  const atlasPath = opts.atlasResultsPath ?? atlasPathFromEnv;

  const buildOpts: { generatedAt?: string; atlasResultsPath?: string } = {};
  if (opts.generatedAt !== undefined) {
    buildOpts.generatedAt = opts.generatedAt;
  }
  if (atlasPath !== undefined) {
    buildOpts.atlasResultsPath = atlasPath;
  }

  const file = buildFrontierFile(rows, buildOpts);

  const outPath = join(resultsDir, "frontier.json");
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(file, null, 2) + "\n", "utf8");
  return file;
}
