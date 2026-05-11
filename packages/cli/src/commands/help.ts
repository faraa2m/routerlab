// commands/help.ts — top-level usage banner.
//
// Renders the multi-subcommand help text. The text is intentionally
// long-form rather than a terse listing because the CLI is the user-facing
// surface for the project and the primary entry point for adoption.

import { type CliContext, writeLine } from "../io.ts";

const HELP_TEXT = `routerlab CLI — cost-quality routing for LLM APIs.

USAGE
  route <subcommand> [flags]

SUBCOMMANDS
  route       Route a single prompt to the cheapest model meeting a quality bar.
  frontier    Pretty-print the published Pareto frontier for a task.
  models      List candidate models (optionally filtered by provider).
  eval        Run eval pipelines. Currently: \`eval frontier --task=<t>\`.
  version     Print CLI + core versions.
  help        Show this message.

EXAMPLES
  # Route a prompt for QA at quality bar 0.85, with a $0.005 budget cap:
  echo "What's the capital of France?" \\
    | route route --task=qa --quality-bar=0.85 --max-cost-usd=0.005

  # Same, but raw JSON for piping into jq:
  route route --task=qa --quality-bar=0.85 --input=prompt.txt --json

  # Look up the frontier for codegen at quality bar 0.85:
  route frontier --task=codegen

  # List candidate models on Anthropic only:
  route models --provider=anthropic

  # Print versions:
  route version

FLAGS (route subcommand)
  --task=<qa|codegen|summarization|classification|reasoning>  (required)
  --quality-bar=<float in [0, 1]>                              (required)
  --input=<path>                                               read prompt from file
  --max-cost-usd=<number>                                      hard budget cap
  --max-latency-ms=<number>                                    hard latency cap
  --json                                                       emit raw RouteDecision JSON

EXIT CODES
  0   success
  1   no candidates pass the filters
  2   invalid input
  3   downstream error (calibration / runner / missing artifact)
`;

export function runHelp(ctx: CliContext): number {
  writeLine(ctx.stdout, HELP_TEXT.trimEnd());
  return 0;
}
