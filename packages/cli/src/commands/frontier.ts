// commands/frontier.ts — read `eval/results/frontier.json` and pretty-print
// the Pareto frontier for a given task.
//
// The frontier file is the public artifact published by the eval harness
// (`eval/frontier/runner.ts`, landing in this same wave). Schema:
//
//   {
//     "schema_version": 1,
//     "generated_at": "<iso8601>",
//     "frontiers": {
//       "<taskClass>": [
//         {
//           "model": "claude-haiku-4-5",
//           "provider": "anthropic",
//           "expectedCost": 0.00042,
//           "expectedQuality": 0.87,
//           "qualityLo95"?: 0.82,
//           "qualityHi95"?: 0.91,
//           "n"?: 12
//         },
//         ...
//       ]
//     }
//   }
//
// We accept either the absolute path via `ROUTERLAB_FRONTIER_PATH` env var
// or the default repo-relative location. Both are validated at read time.

import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { resolve as resolvePath, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { CliError, EXIT_DOWNSTREAM, EXIT_INVALID_INPUT } from "../errors.ts";
import { type CliContext, writeLine } from "../io.ts";
import { parseFormat, parseTask } from "../parse.ts";

// Resolve repo-relative path from this module's location. File lives at
// `packages/cli/src/commands/frontier.ts`; frontier.json is at
// `eval/results/frontier.json` (4 levels up to repo root, then down).
const DEFAULT_FRONTIER_PATH = resolvePath(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "..",
  "..",
  "..",
  "eval",
  "results",
  "frontier.json",
);

const FRONTIER_PATH_ENV_VAR = "ROUTERLAB_FRONTIER_PATH";

/**
 * One row of a published Pareto frontier. The optional CI fields are
 * present when the row came from a measured quality table (post Phase 3);
 * absent when the row was generated from the seeded prior.
 */
export interface FrontierRow {
  model: string;
  provider: string;
  expectedCost: number;
  expectedQuality: number;
  qualityLo95?: number;
  qualityHi95?: number;
  n?: number;
}

/**
 * Parsed shape of the on-disk frontier file. Only the fields we render are
 * required; everything else is tolerated and ignored.
 */
export interface FrontierFile {
  schema_version?: number;
  generated_at?: string;
  frontiers: Partial<Record<string, FrontierRow[]>>;
}

/**
 * Parse and execute the `frontier` subcommand.
 */
export function runFrontier(ctx: CliContext): number {
  let parsed: ReturnType<typeof parseArgs>;
  try {
    parsed = parseArgs({
      args: [...ctx.argv],
      options: {
        task: { type: "string" },
        format: { type: "string" },
        path: { type: "string" },
      },
      strict: true,
      allowPositionals: false,
    });
  } catch (e) {
    throw new CliError(
      EXIT_INVALID_INPUT,
      `frontier: ${e instanceof Error ? e.message : String(e)}`
    );
  }

  const task = parseTask(parsed.values["task"]);
  const format = parseFormat(parsed.values["format"], "table");

  const pathOverride = parsed.values["path"];
  const path = resolveFrontierPath(ctx, typeof pathOverride === "string" ? pathOverride : undefined);

  const file = loadFrontierFile(path);
  const rows = file.frontiers[task];
  if (rows === undefined || rows.length === 0) {
    throw new CliError(
      EXIT_DOWNSTREAM,
      `frontier: no rows for task "${task}" in ${path}`
    );
  }

  if (format === "json") {
    writeLine(
      ctx.stdout,
      JSON.stringify(
        {
          task,
          generated_at: file.generated_at,
          rows,
        },
        null,
        2
      )
    );
    return 0;
  }

  renderTable(task, rows, file, ctx);
  return 0;
}

/**
 * Resolve which frontier.json path to read. Precedence:
 *   1. `--path=<p>` flag.
 *   2. `ROUTERLAB_FRONTIER_PATH` env var.
 *   3. The repo-default absolute path.
 */
function resolveFrontierPath(ctx: CliContext, pathFlag: string | undefined): string {
  if (pathFlag !== undefined && pathFlag.length > 0) {
    return resolvePath(ctx.cwd, pathFlag);
  }
  const fromEnv = ctx.env[FRONTIER_PATH_ENV_VAR];
  if (typeof fromEnv === "string" && fromEnv.length > 0) {
    return fromEnv;
  }
  return DEFAULT_FRONTIER_PATH;
}

function loadFrontierFile(path: string): FrontierFile {
  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch (e) {
    throw new CliError(
      EXIT_DOWNSTREAM,
      `frontier: cannot read ${path}: ${
        e instanceof Error ? e.message : String(e)
      }`
    );
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    throw new CliError(
      EXIT_DOWNSTREAM,
      `frontier: ${path} is not valid JSON: ${
        e instanceof Error ? e.message : String(e)
      }`
    );
  }

  if (parsed === null || typeof parsed !== "object") {
    throw new CliError(
      EXIT_DOWNSTREAM,
      `frontier: ${path} root is not an object`
    );
  }

  const obj = parsed as Record<string, unknown>;
  const frontiers = obj["frontiers"];
  if (frontiers === null || typeof frontiers !== "object") {
    throw new CliError(
      EXIT_DOWNSTREAM,
      `frontier: ${path} missing required "frontiers" object`
    );
  }

  // Schema validation is intentionally shallow — we just verify the rows
  // have the fields we render so a malformed row produces a clean message.
  const normalized: Partial<Record<string, FrontierRow[]>> = {};
  for (const [k, v] of Object.entries(frontiers)) {
    if (!Array.isArray(v)) {
      throw new CliError(
        EXIT_DOWNSTREAM,
        `frontier: ${path} key "frontiers.${k}" is not an array`
      );
    }
    normalized[k] = v.map((row, idx) => validateRow(row, k, idx, path));
  }

  return {
    schema_version: typeof obj["schema_version"] === "number" ? obj["schema_version"] : undefined,
    generated_at: typeof obj["generated_at"] === "string" ? obj["generated_at"] : undefined,
    frontiers: normalized,
  };
}

function validateRow(row: unknown, taskKey: string, idx: number, path: string): FrontierRow {
  if (row === null || typeof row !== "object") {
    throw new CliError(
      EXIT_DOWNSTREAM,
      `frontier: ${path} frontiers.${taskKey}[${idx}] is not an object`
    );
  }
  const r = row as Record<string, unknown>;
  const requireString = (key: string): string => {
    const v = r[key];
    if (typeof v !== "string" || v.length === 0) {
      throw new CliError(
        EXIT_DOWNSTREAM,
        `frontier: ${path} frontiers.${taskKey}[${idx}].${key} must be a non-empty string`
      );
    }
    return v;
  };
  const requireNumber = (key: string): number => {
    const v = r[key];
    if (typeof v !== "number" || !Number.isFinite(v)) {
      throw new CliError(
        EXIT_DOWNSTREAM,
        `frontier: ${path} frontiers.${taskKey}[${idx}].${key} must be a finite number`
      );
    }
    return v;
  };
  const optionalNumber = (key: string): number | undefined => {
    const v = r[key];
    if (v === undefined) return undefined;
    if (typeof v !== "number" || !Number.isFinite(v)) {
      throw new CliError(
        EXIT_DOWNSTREAM,
        `frontier: ${path} frontiers.${taskKey}[${idx}].${key} must be a finite number`
      );
    }
    return v;
  };

  const out: FrontierRow = {
    model: requireString("model"),
    provider: requireString("provider"),
    expectedCost: requireNumber("expectedCost"),
    expectedQuality: requireNumber("expectedQuality"),
  };
  const lo = optionalNumber("qualityLo95");
  if (lo !== undefined) out.qualityLo95 = lo;
  const hi = optionalNumber("qualityHi95");
  if (hi !== undefined) out.qualityHi95 = hi;
  const n = optionalNumber("n");
  if (n !== undefined) out.n = n;
  return out;
}

function renderTable(
  task: string,
  rows: readonly FrontierRow[],
  file: FrontierFile,
  ctx: CliContext
): void {
  const header = `Pareto frontier — task: ${task}`;
  writeLine(ctx.stdout, header);
  writeLine(ctx.stdout, "=".repeat(header.length));
  if (typeof file.generated_at === "string") {
    writeLine(ctx.stdout, `generated_at: ${file.generated_at}`);
  }
  writeLine(ctx.stdout, "");

  // Sort by cost ascending so the cheapest comes first; this is the natural
  // reading order for "what should I pick at quality bar X?" workflows.
  const sorted = [...rows].sort((a, b) => a.expectedCost - b.expectedCost);

  const headers = ["model", "provider", "cost", "quality", "CI95", "n"];
  const data = sorted.map((r) => [
    r.model,
    r.provider,
    `$${r.expectedCost.toFixed(6)}`,
    r.expectedQuality.toFixed(3),
    r.qualityLo95 !== undefined && r.qualityHi95 !== undefined
      ? `${r.qualityLo95.toFixed(3)}..${r.qualityHi95.toFixed(3)}`
      : "—",
    r.n !== undefined ? String(r.n) : "—",
  ]);

  const widths = headers.map((h, col) => {
    const cells = data.map((row) => row[col] ?? "");
    return Math.max(h.length, ...cells.map((s) => s.length));
  });

  const fmt = (cols: readonly string[]): string =>
    cols.map((s, i) => s.padEnd(widths[i] ?? s.length)).join("  ");

  writeLine(ctx.stdout, fmt(headers));
  writeLine(ctx.stdout, fmt(widths.map((w) => "-".repeat(w))));
  for (const row of data) {
    writeLine(ctx.stdout, fmt(row));
  }
}
