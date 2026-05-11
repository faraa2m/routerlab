// eval/frontier/runner.ts — orchestrator for the Pareto-frontier sweep.
//
// Given a set of (task, model) tuples and a sample size, drive the
// per-provider runners, score each completion against the task's
// `score()` function (or, optionally, the LLM judge harness), compute
// atlas-grounded cost via `cost.ts`, persist a `RunOutcome` per call,
// and aggregate the outcomes into per-(task, model) summary rows.
//
// This is the only file in the frontier pipeline that does network I/O.
// `build_frontier.ts`, `quality_table.ts`, and `plot.ts` are pure
// transforms over the artifacts this file produces.
//
// Persistence layout:
//   eval/results/runs/{task}/{model}/{example_id}.json   — RunOutcome
//   eval/results/summary.json                            — SummaryRow[]
//
// Determinism:
//   - Example sampling uses the task's deterministic `loadExamples(seed,
//     limit)`, so the same `--n` + `--seed` selects the same rows.
//   - The orchestrator never shuffles internally; it walks tasks in the
//     order passed in and examples in the order the loader returned.
//   - Output filenames are derived from the example id so re-runs
//     overwrite rather than append.
//
// Strict TS, no `any`, no `@ts-ignore`.

import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import candidatesData from "../../packages/core/src/candidates.json" with { type: "json" };
import {
  estimateCost,
  type CostProvider,
} from "../../packages/core/src/cost.ts";
import type {
  ModelCandidate,
  TaskClass,
} from "../../packages/core/src/types.ts";

import { createRunner, type ProviderId } from "../runners/_factory.ts";
import type { RunResponse, Runner } from "../runners/_types.ts";

import qaTask from "../tasks/qa.ts";
import classificationTask from "../tasks/classification.ts";
import codegenTask from "../tasks/codegen.ts";
import summarizationTask from "../tasks/summarization.ts";
import reasoningTask from "../tasks/reasoning.ts";
import type { TaskDefinition, TaskExample } from "../tasks/_types.ts";

import type {
  RunOutcome,
  SummaryRow,
  TokenSource,
  Confidence,
} from "./_types.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Default per-task example count (per the brief: 50). */
export const DEFAULT_N: number = 50;

/** Smoke mode example count (per the brief: ~5 / 2). We use 2 to match `smoke`. */
export const SMOKE_N: number = 2;

/** Default deterministic seed for example sampling. */
export const DEFAULT_SEED: number = 42;

/** Default output token budget for runner calls. */
const DEFAULT_MAX_TOKENS = 512;

/** Default sampling temperature: low for reproducibility but >0 so models
 * that hate exactly-zero stay happy. */
const DEFAULT_TEMPERATURE = 0.2;

/** Default results root, resolved relative to this file. */
const DEFAULT_RESULTS_DIR: string = (() => {
  // Resolve relative to the eval/frontier directory: ../../eval/results
  const here = dirname(fileURLToPath(import.meta.url));
  return join(here, "..", "results");
})();

// ---------------------------------------------------------------------------
// Task registry
// ---------------------------------------------------------------------------

/**
 * Map of task class → task definition. Keep in lockstep with the
 * `TaskClass` union in `@routerlab/core` — drift here causes silent
 * exclusion from the frontier sweep.
 */
export const TASKS: Readonly<Record<TaskClass, TaskDefinition<unknown, unknown>>> = {
  qa: qaTask as unknown as TaskDefinition<unknown, unknown>,
  classification: classificationTask as unknown as TaskDefinition<unknown, unknown>,
  codegen: codegenTask as unknown as TaskDefinition<unknown, unknown>,
  summarization: summarizationTask as unknown as TaskDefinition<unknown, unknown>,
  reasoning: reasoningTask as unknown as TaskDefinition<unknown, unknown>,
};

const ALL_TASKS: readonly TaskClass[] = [
  "qa",
  "classification",
  "codegen",
  "summarization",
  "reasoning",
];

// ---------------------------------------------------------------------------
// Candidate pool helpers
// ---------------------------------------------------------------------------

interface CandidateFile {
  candidates: ModelCandidate[];
}

const DEFAULT_CANDIDATES: readonly ModelCandidate[] =
  (candidatesData as CandidateFile).candidates;

/**
 * Resolve the candidate pool used by the build orchestrator. Defaults to
 * `candidates.json`. Callers can override (used by tests and by future
 * scoped sweeps).
 */
export function defaultCandidates(): readonly ModelCandidate[] {
  return DEFAULT_CANDIDATES;
}

// ---------------------------------------------------------------------------
// Runner factory abstraction
// ---------------------------------------------------------------------------

/**
 * Signature of a runner factory. Mocked in tests; production wires this
 * to `createRunner` from `eval/runners/_factory.ts`.
 */
export type RunnerFactory = (provider: ProviderId) => Runner;

/** Production-mode runner factory (live API calls). */
export const liveRunnerFactory: RunnerFactory = (p) => createRunner(p);

/**
 * Provider id mapping: routerlab's `Provider` union → factory's
 * `ProviderId` union. The two are equivalent except `openai` / `google`
 * are not in the factory (they aren't wired up yet) — we surface an
 * informative error if the build orchestrator is asked to run one.
 */
function toProviderId(provider: ModelCandidate["provider"]): ProviderId {
  switch (provider) {
    case "anthropic":
    case "groq":
    case "together":
    case "hf":
    case "openrouter":
      return provider;
    case "openai":
    case "google":
      throw new Error(
        `runner not wired up for provider "${provider}" — extend eval/runners/_factory.ts or remove the candidate from candidates.json`,
      );
    default: {
      const _exhaustive: never = provider;
      throw new Error(`unknown provider "${String(_exhaustive)}"`);
    }
  }
}

// ---------------------------------------------------------------------------
// Optional judge hook
// ---------------------------------------------------------------------------

/**
 * Pluggable judge function. When supplied to `runOne`, the build orchestrator
 * delegates quality scoring to the judge instead of calling the task's
 * `score()` function. The sibling `router-judge` agent owns the actual
 * judge implementation; we just accept whatever shape it ends up
 * exporting via a structural function type to keep the dependency loose.
 *
 * The judge is expected to return a normalized score in [0, 1]. Out-of-
 * range values are clamped by `safeClamp01` before persistence.
 */
export type JudgeFn = (req: {
  taskClass: TaskClass;
  prompt: string;
  candidate: string;
  reference: unknown;
}) => Promise<{ score: number }>;

const safeClamp01 = (x: number): number => {
  if (!Number.isFinite(x)) return 0;
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
};

// ---------------------------------------------------------------------------
// Prompt hashing (for audit + idempotency checks)
// ---------------------------------------------------------------------------

/**
 * Deterministic SHA-256 prefix of a prompt. 16 hex chars = 64 bits of
 * entropy — plenty to detect drift between two `loadExamples` runs that
 * are supposed to produce the same rendered prompt.
 */
export function promptDigest(prompt: string): string {
  return createHash("sha256").update(prompt, "utf8").digest("hex").slice(0, 16);
}

// ---------------------------------------------------------------------------
// Per-call run
// ---------------------------------------------------------------------------

export interface RunOneOptions {
  taskClass: TaskClass;
  candidate: ModelCandidate;
  example: TaskExample;
  runner: Runner;
  judge?: JudgeFn;
  /** Override the per-call `maxTokens`. */
  maxTokens?: number;
  /** Override the per-call temperature. */
  temperature?: number;
}

/**
 * Execute one (task, model, example) measurement.
 *
 * The output `RunOutcome` is fully self-describing — the persisted file
 * carries enough metadata that a downstream audit can re-derive the
 * frontier without re-running the eval.
 *
 * On runner error, returns a `RunOutcome` with `error` set; cost is
 * still estimated from the input prompt alone (output tokens default
 * via task-class heuristic). This keeps the audit log uniform.
 */
export async function runOne(opts: RunOneOptions): Promise<RunOutcome> {
  const { taskClass, candidate, example, runner } = opts;
  const task = TASKS[taskClass];
  const prompt = task.promptTemplate(example.input);
  const promptHash = promptDigest(prompt);

  // Cost estimation: always atlas-grounded via cost.ts. We do this BEFORE
  // the runner call so even an errored run carries cost provenance.
  const costEst = estimateCost({
    prompt,
    model: candidate.model,
    provider: candidate.provider as CostProvider,
    pricing: candidate.pricing,
    taskClass,
  });

  let response: RunResponse | undefined;
  let errorMsg: string | undefined;
  try {
    response = await runner.run({
      model: candidate.model,
      prompt,
      maxTokens: opts.maxTokens ?? DEFAULT_MAX_TOKENS,
      temperature: opts.temperature ?? DEFAULT_TEMPERATURE,
    });
  } catch (e) {
    errorMsg = e instanceof Error ? e.message : String(e);
  }

  if (response === undefined || errorMsg !== undefined) {
    return {
      task: taskClass,
      model: candidate.model,
      provider: candidate.provider,
      exampleId: example.id,
      promptHash,
      output: "",
      qualityScore: 0,
      scoreSource: "task-score",
      costUsd: costEst.totalUsd,
      tokenSource: costEst.tokenSource,
      confidence: costEst.confidence,
      latencyMs: 0,
      ts: new Date().toISOString(),
      error: errorMsg ?? "no response from runner",
    };
  }

  // Quality scoring.
  let qualityScore: number;
  let scoreSource: RunOutcome["scoreSource"];
  if (opts.judge !== undefined) {
    const verdict = await opts.judge({
      taskClass,
      prompt,
      candidate: response.output,
      reference: example.reference,
    });
    qualityScore = safeClamp01(verdict.score);
    scoreSource = "judge";
  } else {
    const parsed = task.parseOutput(response.output);
    const rawScore = await task.score(parsed, example.reference);
    qualityScore = safeClamp01(rawScore);
    scoreSource = "task-score";
  }

  // Cost: always grounded through `cost.ts` for atlas-calibration. When
  // the runner reports an output-token count, we refine the estimate with
  // the empirical output length; otherwise the initial heuristic estimate
  // from `costEst` stands. We deliberately do NOT defer to
  // `response.usdCost` (the runner-side cost) — that path uses raw
  // tokenizer output and bypasses the atlas correction factor, which is
  // exactly the differentiator we want recorded in the frontier file.
  let costUsd: number = costEst.totalUsd;
  let finalTokenSource = costEst.tokenSource;
  let finalConfidence = costEst.confidence;
  if (response.outputTokens !== undefined) {
    const refined = estimateCost({
      prompt,
      model: candidate.model,
      provider: candidate.provider as CostProvider,
      pricing: candidate.pricing,
      taskClass,
      expectedOutputTokens: response.outputTokens,
    });
    costUsd = refined.totalUsd;
    finalTokenSource = refined.tokenSource;
    finalConfidence = refined.confidence;
  }

  return {
    task: taskClass,
    model: candidate.model,
    provider: candidate.provider,
    exampleId: example.id,
    promptHash,
    output: response.output,
    qualityScore,
    scoreSource,
    costUsd,
    tokenSource: finalTokenSource,
    confidence: finalConfidence,
    inputTokens: response.inputTokens,
    outputTokens: response.outputTokens,
    latencyMs: response.latencyMs,
    ts: response.ts,
  };
}

// ---------------------------------------------------------------------------
// Persistence helpers
// ---------------------------------------------------------------------------

/**
 * Resolve the on-disk path for one outcome. Sanitizes the model id so
 * the filesystem doesn't choke on `meta-llama/Llama-3.1-…` etc.
 */
export function outcomePath(
  resultsDir: string,
  outcome: Pick<RunOutcome, "task" | "model" | "exampleId">,
): string {
  const safeModel = outcome.model.replace(/[^A-Za-z0-9_.-]/g, "_");
  const safeExample = outcome.exampleId.replace(/[^A-Za-z0-9_.-]/g, "_");
  return join(resultsDir, "runs", outcome.task, safeModel, `${safeExample}.json`);
}

function ensureDir(path: string): void {
  mkdirSync(dirname(path), { recursive: true });
}

/**
 * Write one outcome to disk as pretty-printed JSON. Pretty-print is
 * intentional — these files are small (one per call) and humans grep
 * them when debugging weird scores.
 */
export function writeOutcome(resultsDir: string, outcome: RunOutcome): void {
  const path = outcomePath(resultsDir, outcome);
  ensureDir(path);
  writeFileSync(path, JSON.stringify(outcome, null, 2) + "\n", "utf8");
}

/**
 * Stream all persisted outcomes for a (task, model) bucket from disk.
 * Returns `[]` if the directory does not exist yet.
 */
export function readOutcomes(
  resultsDir: string,
  task: TaskClass,
  model: string,
): RunOutcome[] {
  const safeModel = model.replace(/[^A-Za-z0-9_.-]/g, "_");
  const dir = join(resultsDir, "runs", task, safeModel);
  if (!existsSync(dir)) return [];
  const entries = readdirSync(dir);
  const outcomes: RunOutcome[] = [];
  for (const f of entries) {
    if (!f.endsWith(".json")) continue;
    const raw = readFileSync(join(dir, f), "utf8");
    outcomes.push(JSON.parse(raw) as RunOutcome);
  }
  return outcomes;
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

/**
 * Sorted-copy quantile via linear interpolation. Pure. `p` must be in
 * [0, 1]. Returns 0 for an empty array.
 *
 * Implementation note: this matches numpy's `np.quantile` with the
 * default `linear` method — easy to cross-check from the paper analysis.
 */
export function quantile(values: readonly number[], p: number): number {
  if (values.length === 0) return 0;
  if (p <= 0) return values[0] ?? 0;
  if (p >= 1) return values[values.length - 1] ?? 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const frac = idx - lo;
  const a = sorted[lo] ?? 0;
  const b = sorted[hi] ?? a;
  return a + (b - a) * frac;
}

/**
 * Aggregate a bucket of `RunOutcome` (one task, one model) into a
 * `SummaryRow`. Errored outcomes are excluded from the percentile
 * aggregates and counted separately in `errors`. If every outcome
 * errored, the row is still emitted with `n=0` and zeroed aggregates so
 * callers can see the failure in the summary file.
 */
export function aggregateBucket(
  task: TaskClass,
  outcomes: readonly RunOutcome[],
): SummaryRow {
  if (outcomes.length === 0) {
    throw new Error(`aggregateBucket: empty bucket for task "${task}"`);
  }
  const model = outcomes[0]!.model;
  const provider = outcomes[0]!.provider;
  const ok = outcomes.filter((o) => o.error === undefined);
  const errors = outcomes.length - ok.length;

  const qualities = ok.map((o) => o.qualityScore);
  const costs = ok.map((o) => o.costUsd);

  const mean = (xs: readonly number[]): number => {
    if (xs.length === 0) return 0;
    let s = 0;
    for (const x of xs) s += x;
    return s / xs.length;
  };

  // Token-source / confidence aggregation: pick the modal value across
  // successful outcomes. If everything errored, fall back to the first
  // outcome's attribution (which is still meaningful — cost.ts runs
  // before the runner call).
  const tokenSourceCounts = new Map<TokenSource, number>();
  const confidenceCounts = new Map<Confidence, number>();
  for (const o of ok.length > 0 ? ok : outcomes) {
    tokenSourceCounts.set(o.tokenSource, (tokenSourceCounts.get(o.tokenSource) ?? 0) + 1);
    confidenceCounts.set(o.confidence, (confidenceCounts.get(o.confidence) ?? 0) + 1);
  }
  const modal = <T>(m: Map<T, number>): T => {
    let bestKey: T | undefined;
    let bestCount = -1;
    for (const [k, v] of m.entries()) {
      if (v > bestCount) {
        bestCount = v;
        bestKey = k;
      }
    }
    if (bestKey === undefined) {
      throw new Error("modal: empty map");
    }
    return bestKey;
  };
  const tokenSource: TokenSource = modal(tokenSourceCounts);
  const confidence: Confidence = modal(confidenceCounts);

  // Disagreement note: emit when more than one distinct token-source / confidence
  // value was observed across the bucket.
  const tokenSourceDisagree = tokenSourceCounts.size > 1;
  const confidenceDisagree = confidenceCounts.size > 1;
  const note: string[] = [];
  if (tokenSourceDisagree) {
    const breakdown = [...tokenSourceCounts.entries()]
      .map(([k, v]) => `${k}=${v}`)
      .join(", ");
    note.push(`token-source mixed: ${breakdown}; reporting modal value`);
  }
  if (confidenceDisagree) {
    const breakdown = [...confidenceCounts.entries()]
      .map(([k, v]) => `${k}=${v}`)
      .join(", ");
    note.push(`confidence mixed: ${breakdown}; reporting modal value`);
  }

  const row: SummaryRow = {
    task,
    model,
    provider,
    n: ok.length,
    mean_quality: mean(qualities),
    mean_cost_usd: mean(costs),
    p50_quality: quantile(qualities, 0.5),
    p50_cost_usd: quantile(costs, 0.5),
    p95_quality: quantile(qualities, 0.95),
    p95_cost_usd: quantile(costs, 0.95),
    tokenSource,
    confidence,
    errors,
  };
  if (note.length > 0) {
    row.tokenSourceNote = note.join("; ");
  }
  return row;
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

export interface RunFrontierOptions {
  /** Which tasks to run. Defaults to all five. */
  tasks?: readonly TaskClass[];
  /** Candidate pool to evaluate. Defaults to `candidates.json`. */
  candidates?: readonly ModelCandidate[];
  /** Examples per (task, model). Default `DEFAULT_N`. */
  examplesPerTask?: number;
  /** Seed for example sampling. Default `DEFAULT_SEED`. */
  seed?: number;
  /** Override runner factory (tests inject mocks). */
  runnerFactory?: RunnerFactory;
  /** Optional judge override. When set, every score uses the judge. */
  judge?: JudgeFn;
  /** Override the results dir. Default: `eval/results`. */
  resultsDir?: string;
  /** Suppress per-call console logging (tests set this). */
  quiet?: boolean;
  /**
   * If true, skip any (task, model) bucket that already has at least
   * `examplesPerTask` outcomes persisted. Lets the build orchestrator be
   * interrupted and resumed without re-spending API budget. Default false.
   */
  resume?: boolean;
}

export interface RunFrontierResult {
  summary: SummaryRow[];
  /**
   * Per-(task, model) outcomes. Useful for tests that don't want to
   * round-trip through disk; production code can just read summary.json.
   */
  outcomes: RunOutcome[];
}

/**
 * Drive the full sweep. The result is the per-(task, model) summary
 * array; raw outcomes are persisted to disk and also returned in-memory.
 *
 * Behavior:
 *   - For each task in `tasks`:
 *     - Load `examplesPerTask` examples (deterministic via seed).
 *     - For each candidate in `candidates`:
 *       - Construct (or reuse) a runner via `runnerFactory`.
 *       - Optionally skip if `resume` and enough outcomes already exist.
 *       - For each example: call `runOne`, persist the outcome.
 *       - Aggregate to a `SummaryRow`.
 *   - Persist `summary.json` and return.
 *
 * Robustness:
 *   - A failing runner construction (e.g. missing env var) is treated as
 *     "skip this provider for the whole sweep" with a log line — keeps
 *     mixed-credentialled runs viable for the cross-provider demo.
 *   - A failing per-call run is captured as `RunOutcome.error` and
 *     doesn't abort the bucket.
 */
export async function runFrontier(
  opts: RunFrontierOptions = {},
): Promise<RunFrontierResult> {
  const tasks = opts.tasks ?? ALL_TASKS;
  const candidates = opts.candidates ?? DEFAULT_CANDIDATES;
  const examplesPerTask = opts.examplesPerTask ?? DEFAULT_N;
  const seed = opts.seed ?? DEFAULT_SEED;
  const factory = opts.runnerFactory ?? liveRunnerFactory;
  const resultsDir = opts.resultsDir ?? DEFAULT_RESULTS_DIR;
  const quiet = opts.quiet ?? false;
  const resume = opts.resume ?? false;

  // Cache runners per provider (constructing them validates env vars).
  const runners = new Map<string, Runner | "skip">();
  const getRunner = (cand: ModelCandidate): Runner | undefined => {
    const cached = runners.get(cand.provider);
    if (cached !== undefined) return cached === "skip" ? undefined : cached;
    let providerId: ProviderId;
    try {
      providerId = toProviderId(cand.provider);
    } catch (e) {
      if (!quiet) {
        console.warn(
          `[frontier] skipping provider "${cand.provider}": ${
            e instanceof Error ? e.message : String(e)
          }`,
        );
      }
      runners.set(cand.provider, "skip");
      return undefined;
    }
    try {
      const r = factory(providerId);
      runners.set(cand.provider, r);
      return r;
    } catch (e) {
      if (!quiet) {
        console.warn(
          `[frontier] skipping provider "${cand.provider}" (factory error): ${
            e instanceof Error ? e.message : String(e)
          }`,
        );
      }
      runners.set(cand.provider, "skip");
      return undefined;
    }
  };

  const allOutcomes: RunOutcome[] = [];
  const summary: SummaryRow[] = [];

  for (const taskClass of tasks) {
    const task = TASKS[taskClass];
    const examples = await task.loadExamples({ limit: examplesPerTask, seed });

    if (examples.length === 0) {
      if (!quiet) {
        console.warn(`[frontier] task "${taskClass}" produced 0 examples — skipping`);
      }
      continue;
    }

    for (const cand of candidates) {
      // Resume short-circuit.
      if (resume) {
        const existing = readOutcomes(resultsDir, taskClass, cand.model);
        if (existing.length >= examples.length) {
          if (!quiet) {
            console.log(
              `[frontier] resume: ${taskClass}/${cand.model} already has ${existing.length}/${examples.length} outcomes — using on disk`,
            );
          }
          allOutcomes.push(...existing);
          summary.push(aggregateBucket(taskClass, existing));
          continue;
        }
      }

      const runner = getRunner(cand);
      const bucket: RunOutcome[] = [];
      for (const ex of examples) {
        let outcome: RunOutcome;
        if (runner === undefined) {
          // Provider skipped — record a synthetic errored outcome so the
          // summary file still has a row for this (task, model) bucket.
          // Cost is still atlas-grounded.
          const prompt = task.promptTemplate(ex.input);
          const costEst = estimateCost({
            prompt,
            model: cand.model,
            provider: cand.provider as CostProvider,
            pricing: cand.pricing,
            taskClass,
          });
          outcome = {
            task: taskClass,
            model: cand.model,
            provider: cand.provider,
            exampleId: ex.id,
            promptHash: promptDigest(prompt),
            output: "",
            qualityScore: 0,
            scoreSource: "task-score",
            costUsd: costEst.totalUsd,
            tokenSource: costEst.tokenSource,
            confidence: costEst.confidence,
            latencyMs: 0,
            ts: new Date().toISOString(),
            error: `runner unavailable for provider "${cand.provider}"`,
          };
        } else {
          outcome = await runOne({
            taskClass,
            candidate: cand,
            example: ex,
            runner,
            ...(opts.judge !== undefined ? { judge: opts.judge } : {}),
          });
        }
        writeOutcome(resultsDir, outcome);
        bucket.push(outcome);
        allOutcomes.push(outcome);
        if (!quiet) {
          const tag = outcome.error !== undefined ? "ERR " : "OK  ";
          console.log(
            `[frontier] ${tag}${taskClass}/${cand.model}/${ex.id} ` +
              `q=${outcome.qualityScore.toFixed(3)} cost=$${outcome.costUsd.toFixed(6)} ` +
              `src=${outcome.tokenSource}${outcome.error !== undefined ? ` err=${outcome.error.slice(0, 80)}` : ""}`,
          );
        }
      }
      summary.push(aggregateBucket(taskClass, bucket));
    }
  }

  // Persist summary.json.
  const summaryPath = join(resultsDir, "summary.json");
  ensureDir(summaryPath);
  writeFileSync(
    summaryPath,
    JSON.stringify(
      {
        schema_version: 1,
        generated_at: new Date().toISOString(),
        rows: summary,
      },
      null,
      2,
    ) + "\n",
    "utf8",
  );

  return { summary, outcomes: allOutcomes };
}

/**
 * Read a previously persisted summary file. Returns `null` if the file is
 * missing (callers decide whether to run the sweep on demand).
 */
export function readSummary(
  resultsDir: string = DEFAULT_RESULTS_DIR,
): SummaryRow[] | null {
  const path = join(resultsDir, "summary.json");
  if (!existsSync(path)) return null;
  const raw = readFileSync(path, "utf8");
  const parsed = JSON.parse(raw) as { rows?: SummaryRow[] };
  return parsed.rows ?? null;
}

export const __TEST_DEFAULT_RESULTS_DIR: string = DEFAULT_RESULTS_DIR;
