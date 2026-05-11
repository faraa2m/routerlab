// commands/eval.ts — the `eval` namespace.
//
// Currently exposes one sub-subcommand: `eval frontier --task=<t> --n=<n>`.
// The brief says this should delegate to `eval/frontier/runner.ts` either
// via subprocess or direct import.
//
// At the moment the frontier runner has not been published yet by the
// router-frontier agent; rather than failing in a hostile way, we attempt
// a dynamic import of the runner module by its conventional path. If the
// module is present we invoke its `runFrontier({ task, n })` export; if
// not, we exit 3 with a clear "not yet implemented in this build" message
// so the user knows what's going on.
//
// This design keeps the CLI shippable now and "just works" the moment the
// runner agent lands its module — no further CLI changes needed.

import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";
import { dirname, resolve as resolvePath } from "node:path";

import { CliError, EXIT_DOWNSTREAM, EXIT_INVALID_INPUT } from "../errors.ts";
import { type CliContext, writeLine } from "../io.ts";
import { parsePositiveIntOptional, parseTask } from "../parse.ts";

const FRONTIER_RUNNER_MODULE_ENV_VAR = "ROUTERLAB_FRONTIER_RUNNER_MODULE";

// Resolve repo-relative path from this module's location so the CLI works on
// any machine without hardcoded absolute paths. This file lives at
// `packages/cli/src/commands/eval.ts`; the runner is at
// `eval/frontier/runner.ts` relative to the repo root (4 levels up).
const DEFAULT_FRONTIER_RUNNER_MODULE = resolvePath(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "..",
  "..",
  "..",
  "eval",
  "frontier",
  "runner.ts",
);

/**
 * Minimal contract the dynamically-loaded frontier runner must satisfy.
 * Matches the public surface of `eval/frontier/runner.ts`'s `runFrontier`.
 * Kept structurally typed (rather than importing the real type) so the
 * CLI doesn't take a hard dependency on the eval harness — the runner
 * lives outside of `@routerlab/core`'s public surface.
 */
export interface FrontierRunnerModule {
  runFrontier: (opts: {
    tasks?: readonly string[];
    examplesPerTask?: number;
    quiet?: boolean;
  }) => Promise<unknown> | unknown;
}

/**
 * Entry point for the `eval ...` namespace. The first positional after
 * `eval` selects the sub-subcommand.
 */
export async function runEval(ctx: CliContext): Promise<number> {
  const [first, ...rest] = ctx.argv;
  if (first === undefined) {
    throw new CliError(
      EXIT_INVALID_INPUT,
      "eval: missing subcommand. Usage: route eval frontier --task=<t> [--n=<n>]"
    );
  }
  const subCtx: CliContext = { ...ctx, argv: rest };
  switch (first) {
    case "frontier":
      return runEvalFrontier(subCtx);
    default:
      throw new CliError(
        EXIT_INVALID_INPUT,
        `eval: unknown subcommand "${first}". Known: frontier`
      );
  }
}

/**
 * `eval frontier --task=<task> [--n=<n>]` — invoke the frontier builder.
 */
async function runEvalFrontier(ctx: CliContext): Promise<number> {
  let parsed: ReturnType<typeof parseArgs>;
  try {
    parsed = parseArgs({
      args: [...ctx.argv],
      options: {
        task: { type: "string" },
        n: { type: "string" },
      },
      strict: true,
      allowPositionals: false,
    });
  } catch (e) {
    throw new CliError(
      EXIT_INVALID_INPUT,
      `eval frontier: ${e instanceof Error ? e.message : String(e)}`
    );
  }

  const task = parseTask(parsed.values["task"]);
  const n = parsePositiveIntOptional("n", parsed.values["n"]);

  const modulePath = resolveRunnerModule(ctx);
  let mod: FrontierRunnerModule;
  try {
    const imported = (await import(modulePath)) as Partial<FrontierRunnerModule>;
    if (typeof imported.runFrontier !== "function") {
      throw new Error(
        `module at ${modulePath} does not export a "runFrontier" function`
      );
    }
    mod = imported as FrontierRunnerModule;
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new CliError(
      EXIT_DOWNSTREAM,
      `eval frontier: frontier runner is not available at ${modulePath}: ${msg}. ` +
        `Set ${FRONTIER_RUNNER_MODULE_ENV_VAR} to override the runner path.`
    );
  }

  writeLine(
    ctx.stderr,
    `eval frontier: invoking ${modulePath} (task=${task}${n !== undefined ? `, examplesPerTask=${n}` : ""})`
  );
  const opts: { tasks: readonly string[]; examplesPerTask?: number; quiet: boolean } = {
    tasks: [task],
    quiet: true,
    ...(n !== undefined ? { examplesPerTask: n } : {}),
  };
  const result = await Promise.resolve(mod.runFrontier(opts));
  writeLine(
    ctx.stdout,
    JSON.stringify({ task, n, result: result ?? null }, null, 2)
  );
  return 0;
}

function resolveRunnerModule(ctx: CliContext): string {
  const fromEnv = ctx.env[FRONTIER_RUNNER_MODULE_ENV_VAR];
  if (typeof fromEnv === "string" && fromEnv.length > 0) return fromEnv;
  return DEFAULT_FRONTIER_RUNNER_MODULE;
}
