#!/usr/bin/env bun
// eval/cli.ts — Bun CLI for the frontier pipeline.
//
// Usage:
//   bun eval/cli.ts frontier --task=qa --n=50
//   bun eval/cli.ts frontier-all --n=50
//   bun eval/cli.ts smoke                          # n=2 fixture-only sanity run
//   bun eval/cli.ts build-frontier                 # rebuild frontier.json from summary.json
//   bun eval/cli.ts build-quality-table            # derive quality_table.json
//   bun eval/cli.ts plot                           # render SVGs from frontier.json
//
// Smoke mode is the fast / no-API-keys path: it uses mocked runners and a
// hand-rolled fixture pool so the pipeline can be exercised end-to-end on
// a machine with zero secrets. Useful for CI and PR review.
//
// The orchestrator commands (`frontier`, `frontier-all`) will skip any
// provider whose credentials are missing — running with a partial keyring
// is supported (you get fewer rows in `summary.json`, not an abort).

import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

import type { TaskClass } from "../packages/core/src/types.ts";

import { buildFrontierFromDisk } from "./frontier/build_frontier.ts";
import { buildQualityTableFromDisk } from "./frontier/quality_table.ts";
import { plotAllFromDisk } from "./frontier/plot.ts";
import {
  defaultCandidates,
  DEFAULT_N,
  runFrontier,
  SMOKE_N,
  TASKS,
  type JudgeFn,
  type RunFrontierOptions,
} from "./frontier/runner.ts";
import type { RunResponse, Runner } from "./runners/_types.ts";
import type { ProviderId } from "./runners/_factory.ts";
import { datasetCachePath } from "./tasks/_types.ts";

// ---------------------------------------------------------------------------
// Argv parsing
// ---------------------------------------------------------------------------

interface ParsedArgs {
  command: string;
  flags: Map<string, string | boolean>;
}

function parseArgs(argv: readonly string[]): ParsedArgs {
  // argv layout (under Bun): [runtime, script, command, --flag=value, ...]
  const command = argv[2] ?? "";
  const flags = new Map<string, string | boolean>();
  for (let i = 3; i < argv.length; i++) {
    const arg = argv[i]!;
    if (!arg.startsWith("--")) continue;
    const eq = arg.indexOf("=");
    if (eq === -1) {
      flags.set(arg.slice(2), true);
    } else {
      const k = arg.slice(2, eq);
      const v = arg.slice(eq + 1);
      flags.set(k, v);
    }
  }
  return { command, flags };
}

function strFlag(flags: ParsedArgs["flags"], key: string): string | undefined {
  const v = flags.get(key);
  if (v === undefined) return undefined;
  if (typeof v === "boolean") return "";
  return v;
}

function numFlag(flags: ParsedArgs["flags"], key: string): number | undefined {
  const v = strFlag(flags, key);
  if (v === undefined || v === "") return undefined;
  const n = Number(v);
  if (!Number.isFinite(n)) return undefined;
  return n;
}

function boolFlag(flags: ParsedArgs["flags"], key: string): boolean {
  const v = flags.get(key);
  if (v === undefined) return false;
  if (v === true) return true;
  // Treat `--flag=false` / `--flag=0` as false.
  return !(v === "false" || v === "0");
}

// ---------------------------------------------------------------------------
// Task resolution
// ---------------------------------------------------------------------------

function resolveTask(s: string): TaskClass {
  if (!(s in TASKS)) {
    throw new Error(
      `unknown task "${s}". valid: ${Object.keys(TASKS).join(", ")}`,
    );
  }
  return s as TaskClass;
}

// ---------------------------------------------------------------------------
// Smoke-mode mocked runner
// ---------------------------------------------------------------------------
//
// The smoke command runs the build orchestrator end-to-end with no API keys
// using a synthetic runner that produces deterministic outputs. The
// outputs are crafted to hit each task's reference shape so the scoring
// path runs through the same `parseOutput`/`score` it would in
// production.

function fixtureRunner(provider: ProviderId, modelHint: string): Runner {
  // Synthetic outputs per task class. These are recognized by each
  // task's parser/scorer so smoke can exercise the full pipeline.
  return {
    provider,
    listModels: () => [modelHint],
    async run(req): Promise<RunResponse> {
      const prompt = req.prompt;
      const lower = prompt.toLowerCase();
      let output: string;
      if (lower.includes("classify this tweet")) {
        output = "neutral";
      } else if (lower.includes("complete the following python function")) {
        output = "    return 0\n";
      } else if (lower.includes("solve this problem step by step")) {
        output = "Step 1: Add.\n#### 0";
      } else if (lower.includes("summarize the following article")) {
        output = "A short fixture summary.";
      } else {
        // Default: extractive-QA-style answer.
        output = "fixture answer";
      }
      // Deterministic tiny token counts (no real provider calls).
      return {
        model: req.model,
        output,
        inputTokens: 16,
        outputTokens: 8,
        usdCost: 0,
        latencyMs: 1,
        ts: "1970-01-01T00:00:00.000Z",
      };
    },
  };
}

function smokeRunnerFactory(provider: ProviderId): Runner {
  // The candidate pool uses short model ids; surface the hint so the
  // smoke logs are interpretable.
  return fixtureRunner(provider, `smoke-${provider}-fixture`);
}

function seedSmokeFixtures(): void {
  const fixtures: Record<string, string> = {
    qa: "qa.jsonl",
    classification: "classification.jsonl",
    summarization: "summarization.jsonl",
    reasoning: "reasoning.jsonl",
  };

  for (const [task, filename] of Object.entries(fixtures)) {
    const source = new URL(`./tasks/fixtures/${filename}`, import.meta.url).pathname;
    const target = datasetCachePath(filename);
    if (!existsSync(source)) {
      throw new Error(`missing smoke fixture for ${task}: ${source}`);
    }
    mkdirSync(dirname(target), { recursive: true });
    copyFileSync(source, target);
  }
}

// ---------------------------------------------------------------------------
// Command dispatch
// ---------------------------------------------------------------------------

function helpText(): string {
  return [
    "routerlab frontier pipeline",
    "",
    "usage:",
    "  bun eval/cli.ts frontier --task=<task> [--n=50] [--seed=42] [--resume]",
    "  bun eval/cli.ts frontier-all [--n=50] [--seed=42] [--resume]",
    "  bun eval/cli.ts smoke",
    "  bun eval/cli.ts build-frontier",
    "  bun eval/cli.ts build-quality-table",
    "  bun eval/cli.ts plot",
    "",
    "tasks: " + Object.keys(TASKS).join(", "),
    "",
    "outputs:",
    "  eval/results/runs/{task}/{model}/{example_id}.json   per-call audit log",
    "  eval/results/summary.json                            per-(task, model) aggregates",
    "  eval/results/frontier.json                           per-task Pareto frontier",
    "  eval/results/quality_table.json                      predictor-consumable",
    "  eval/results/plots/<task>.svg                        per-task scatter plots",
    "",
    "  No API calls in `smoke`; all other commands hit live provider APIs.",
  ].join("\n");
}

interface CommandContext {
  args: ParsedArgs;
  resultsDir: string;
}

async function cmdFrontier(ctx: CommandContext): Promise<number> {
  const { args } = ctx;
  const taskStr = strFlag(args.flags, "task");
  if (taskStr === undefined || taskStr === "") {
    console.error("frontier: --task=<task> is required");
    return 2;
  }
  const task = resolveTask(taskStr);
  const n = numFlag(args.flags, "n") ?? DEFAULT_N;
  const seed = numFlag(args.flags, "seed");
  const resume = boolFlag(args.flags, "resume");

  const opts: RunFrontierOptions = {
    tasks: [task],
    examplesPerTask: n,
    resultsDir: ctx.resultsDir,
    resume,
  };
  if (seed !== undefined) opts.seed = seed;

  const { summary } = await runFrontier(opts);
  console.log(`frontier: wrote ${summary.length} summary rows for "${task}"`);
  buildFrontierFromDisk({ resultsDir: ctx.resultsDir });
  buildQualityTableFromDisk({ resultsDir: ctx.resultsDir });
  plotAllFromDisk({ resultsDir: ctx.resultsDir });
  console.log("frontier: wrote frontier.json, quality_table.json, plots/");
  return 0;
}

async function cmdFrontierAll(ctx: CommandContext): Promise<number> {
  const { args } = ctx;
  const n = numFlag(args.flags, "n") ?? DEFAULT_N;
  const seed = numFlag(args.flags, "seed");
  const resume = boolFlag(args.flags, "resume");

  const opts: RunFrontierOptions = {
    examplesPerTask: n,
    resultsDir: ctx.resultsDir,
    resume,
  };
  if (seed !== undefined) opts.seed = seed;

  const { summary } = await runFrontier(opts);
  console.log(`frontier-all: wrote ${summary.length} summary rows`);
  buildFrontierFromDisk({ resultsDir: ctx.resultsDir });
  buildQualityTableFromDisk({ resultsDir: ctx.resultsDir });
  plotAllFromDisk({ resultsDir: ctx.resultsDir });
  console.log("frontier-all: wrote frontier.json, quality_table.json, plots/");
  return 0;
}

async function cmdSmoke(ctx: CommandContext): Promise<number> {
  // Smoke uses fixture runners + the full candidate pool but only runs
  // examples for the lightweight tasks (skip codegen — it spawns python
  // subprocesses that, while sandboxed, take time and have a real OS
  // dependency we don't want in CI smoke).
  const tasks: TaskClass[] = ["qa", "classification", "summarization", "reasoning"];
  seedSmokeFixtures();
  const { summary } = await runFrontier({
    tasks,
    examplesPerTask: SMOKE_N,
    resultsDir: ctx.resultsDir,
    runnerFactory: smokeRunnerFactory,
    quiet: false,
  });
  console.log(`smoke: wrote ${summary.length} summary rows`);
  buildFrontierFromDisk({ resultsDir: ctx.resultsDir });
  buildQualityTableFromDisk({ resultsDir: ctx.resultsDir });
  plotAllFromDisk({ resultsDir: ctx.resultsDir });
  console.log("smoke: wrote frontier.json, quality_table.json, plots/");
  return 0;
}

function cmdBuildFrontier(ctx: CommandContext): number {
  buildFrontierFromDisk({ resultsDir: ctx.resultsDir });
  console.log(`build-frontier: wrote ${ctx.resultsDir}/frontier.json`);
  return 0;
}

function cmdBuildQualityTable(ctx: CommandContext): number {
  buildQualityTableFromDisk({ resultsDir: ctx.resultsDir });
  console.log(`build-quality-table: wrote ${ctx.resultsDir}/quality_table.json`);
  return 0;
}

function cmdPlot(ctx: CommandContext): number {
  const paths = plotAllFromDisk({ resultsDir: ctx.resultsDir });
  for (const p of paths) console.log(`plot: wrote ${p}`);
  return 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

// Silence unused-import warning for fs helpers used only conditionally.
void existsSync;
void readFileSync;
void writeFileSync;
void join;
// `JudgeFn` type re-export anchor (kept available for future plumbing).
const _judgeFnAnchor: JudgeFn | undefined = undefined;
void _judgeFnAnchor;

async function main(): Promise<number> {
  const args = parseArgs(process.argv);
  const resultsDir =
    strFlag(args.flags, "results-dir") ?? join(process.cwd(), "eval", "results");
  const ctx: CommandContext = { args, resultsDir };

  switch (args.command) {
    case "frontier":
      return cmdFrontier(ctx);
    case "frontier-all":
      return cmdFrontierAll(ctx);
    case "smoke":
      return cmdSmoke(ctx);
    case "build-frontier":
      return cmdBuildFrontier(ctx);
    case "build-quality-table":
      return cmdBuildQualityTable(ctx);
    case "plot":
      return cmdPlot(ctx);
    case "":
    case "help":
    case "--help":
    case "-h":
      console.log(helpText());
      return 0;
    default:
      console.error(`unknown command: ${args.command}\n`);
      console.error(helpText());
      return 2;
  }
}

// Bun-friendly main detection.
const isMain = (() => {
  try {
    const meta = import.meta as ImportMeta & { main?: boolean };
    if (typeof meta.main === "boolean") return meta.main;
  } catch {
    /* fall through */
  }
  return process.argv[1]?.endsWith("cli.ts") ?? false;
})();

if (isMain) {
  main()
    .then((code) => process.exit(code))
    .catch((err: unknown) => {
      console.error("fatal:", err);
      process.exit(1);
    });
}
