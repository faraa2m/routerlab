// eval/tasks/_types.ts — shared types for routerlab eval task definitions.
//
// Every task class routerlab routes for (qa, classification, codegen,
// summarization, reasoning) ships one file here that exports a default
// `TaskDefinition`. Runners under `eval/runners/` consume these tasks
// uniformly: load examples, render the prompt, call the model, parse,
// score. The router engine then aggregates per-(task, model) scores into
// a Pareto frontier in `eval/results/frontier.json`.
//
// The contract is intentionally narrow:
//   - examples are loaded lazily and cached on disk
//   - the prompt template is a pure string-rendering function
//   - parse + score are pure and deterministic
//   - score is bounded to [0, 1] so we can compose across tasks
//
// No `any`. No `@ts-ignore`. Strict TS only.

/**
 * One labelled example for a task. `input` is a flat dict of template
 * variables the task's prompt template will interpolate. `reference`
 * holds the ground truth in whatever shape the task's scoring function
 * expects (a string, a list of strings, a label name, etc.).
 *
 * `metadata` is a free-form bag for provenance (dataset row id, split,
 * answer-is-possible flag, etc.). Scoring must not depend on it.
 */
export interface TaskExample {
  id: string;
  input: Record<string, string>;
  reference: unknown;
  metadata?: Record<string, unknown>;
}

/**
 * Options accepted by every `loadExamples` implementation.
 * `limit` and `seed` together guarantee deterministic subsets across runs.
 *
 * When `seed` is omitted, a fixed default (see implementations) is used so
 * that re-running without explicitly setting a seed still reproduces.
 */
export interface LoadExamplesOptions {
  limit?: number;
  seed?: number;
}

/**
 * A full task definition. Generic over the ground-truth `RefType` and the
 * parsed-output `OutputType` so each task's `parseOutput` and `score` get
 * tight types instead of the loose `unknown` carried in `TaskExample`.
 *
 * Convention: `name` matches the `TaskClass` enum in
 * `packages/core/src/types.ts`. Drift between the two will break frontier
 * publication, so keep them in sync.
 */
export interface TaskDefinition<RefType, OutputType> {
  /** Stable identifier — matches `TaskClass` in `@routerlab/core`. */
  name: string;
  /** One-line human description used in CLI listings / paper tables. */
  description: string;
  /** Pure render: `input` -> prompt string. Must not access env / IO. */
  promptTemplate: (input: TaskExample["input"]) => string;
  /** Lazy + cached. Implementations cache to `.cache/eval-datasets/`. */
  loadExamples: (opts?: LoadExamplesOptions) => Promise<TaskExample[]>;
  /** Pure parse of a raw model completion into the task's output type. */
  parseOutput: (raw: string) => OutputType;
  /**
   * Score in [0, 1]; 1.0 = perfect, 0.0 = wrong.
   *
   * Allowed to return a `Promise<number>` so tasks that genuinely need IO
   * (e.g. codegen, which executes hidden test scripts in a subprocess) can
   * still satisfy the contract. Tasks that don't need IO (qa,
   * classification, summarization, reasoning) return a plain number — both
   * are valid and runners must `await` defensively.
   */
  score: (parsed: OutputType, reference: RefType) => number | Promise<number>;
}

/**
 * Tiny seeded PRNG (mulberry32). Pure function, no allocations beyond the
 * returned closure. Used by every task's `loadExamples` to do deterministic
 * shuffles without pulling in `seedrandom` or similar.
 *
 * Reference: https://stackoverflow.com/a/47593316 (mulberry32, public domain).
 */
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Deterministic Fisher-Yates shuffle, seeded via `mulberry32`. Returns a
 * fresh array; does not mutate the input. Used in `loadExamples` to pick
 * a reproducible subset of a large dataset.
 */
export function seededShuffle<T>(items: readonly T[], seed: number): T[] {
  const rng = mulberry32(seed);
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    // Type system is satisfied because i, j are in-bounds by construction;
    // `noUncheckedIndexedAccess` requires the non-null assertion here.
    const a = out[i]!;
    const b = out[j]!;
    out[i] = b;
    out[j] = a;
  }
  return out;
}

/**
 * Resolve a path under `<repo>/.cache/eval-datasets/`. Centralized so all
 * tasks cache to the same place and `.gitignore` already covers `.cache/`.
 */
export function datasetCachePath(filename: string): string {
  // Resolve relative to the eval/tasks/ directory at runtime. We can't use
  // import.meta.dir in a way the linter likes for all bundlers, so we
  // anchor on process.cwd() + a known relative segment when present, else
  // import.meta.url. Bun supports both; we prefer the URL form for purity.
  const here = new URL("../../.cache/eval-datasets/", import.meta.url);
  return new URL(filename, here).pathname;
}
