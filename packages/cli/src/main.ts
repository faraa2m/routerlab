// main.ts — top-level CLI dispatcher.
//
// Reads the first argv positional, routes it to the matching subcommand
// module, and converts thrown errors into exit codes. Tests import this
// function directly (rather than spawning a subprocess) by passing a
// `CliContext` with in-memory streams.
//
// Exit codes are defined in `./errors.ts` and mirrored in the CLI README.

import { CliError, EXIT_DOWNSTREAM, EXIT_INVALID_INPUT } from "./errors.ts";
import { runEval } from "./commands/eval.ts";
import { runFrontier } from "./commands/frontier.ts";
import { runHelp } from "./commands/help.ts";
import { runModels } from "./commands/models.ts";
import { runRoute } from "./commands/route.ts";
import { runVersion } from "./commands/version.ts";
import { type CliContext, writeLine } from "./io.ts";

/**
 * Top-level entry point. Always resolves with an exit code; never throws.
 *
 * Lifecycle:
 *   1. Extract first positional as the subcommand name.
 *   2. Dispatch to that subcommand's `run*` function with a shifted argv.
 *   3. Any `CliError` thrown by a subcommand is rendered to stderr with
 *      its embedded exit code returned verbatim.
 *   4. Any other exception is mapped to `EXIT_DOWNSTREAM` with the
 *      original message preserved.
 */
export async function main(ctx: CliContext): Promise<number> {
  const [first, ...rest] = ctx.argv;

  // No subcommand → show help and exit 0. This is the same behavior as
  // `git` (interactive shells) and is friendlier than a usage error.
  if (first === undefined || first === "--help" || first === "-h" || first === "help") {
    return runHelp({ ...ctx, argv: rest });
  }

  const subCtx: CliContext = { ...ctx, argv: rest };

  try {
    switch (first) {
      case "route":
        return await runRoute(subCtx);
      case "frontier":
        return runFrontier(subCtx);
      case "models":
        return runModels(subCtx);
      case "eval":
        return await runEval(subCtx);
      case "version":
      case "--version":
      case "-v":
        return runVersion(subCtx);
      default:
        throw new CliError(
          EXIT_INVALID_INPUT,
          `unknown subcommand "${first}". Run \`route help\` for usage.`
        );
    }
  } catch (e) {
    if (e instanceof CliError) {
      writeLine(ctx.stderr, `error: ${e.message}`);
      return e.code;
    }
    const msg = e instanceof Error ? e.message : String(e);
    writeLine(ctx.stderr, `error: ${msg}`);
    return EXIT_DOWNSTREAM;
  }
}
