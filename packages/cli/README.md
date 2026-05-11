# `@routerlab/cli`

> `route` — the terminal interface to routerlab's cost-quality routing engine.

The CLI lets you route prompts, inspect published Pareto frontiers, list
candidate models, and run the frontier eval pipeline — all from a shell.
It wraps `@routerlab/core` so every routing decision is grounded in the
same atlas-calibrated empirical token economics that powers the library.

## Install

```bash
bun add @routerlab/cli
```

Or, inside the routerlab monorepo:

```bash
bun install
bun packages/cli/src/index.ts --help
```

## Quick examples

```bash
# Route a prompt for QA at quality bar 0.85:
echo "What's the capital of France?" \
  | route route --task=qa --quality-bar=0.85

# Same, with a $0.005 hard budget and JSON output for piping into jq:
echo "Write a python function that ..." \
  | route route --task=codegen --quality-bar=0.80 --max-cost-usd=0.005 --json \
  | jq '.chosen.model.model'

# Read the prompt from a file instead of stdin:
route route --task=qa --quality-bar=0.85 --input=./prompt.txt

# Inspect the published Pareto frontier for codegen:
route frontier --task=codegen

# Same, as JSON:
route frontier --task=codegen --format=json

# List candidate models (optionally filtered by provider):
route models
route models --provider=anthropic
route models --json

# Run the frontier eval pipeline for a single task:
route eval frontier --task=qa --n=20

# Print versions:
route version
```

## Subcommands

### `route route --task=<t> --quality-bar=<q> [flags]`

Reads a prompt from `--input <file>` or stdin and prints the engine's
decision: the chosen model, ordered fallbacks, and a skipped-with-reasons
list. Default output is human-friendly; `--json` emits the raw
`RouteDecision` for piping.

| Flag                | Required | Description                                            |
| ------------------- | -------- | ------------------------------------------------------ |
| `--task=<t>`        | yes      | One of `qa, codegen, summarization, classification, reasoning` |
| `--quality-bar=<q>` | yes      | Float in [0, 1]                                        |
| `--input=<path>`    | no       | Read prompt from file (else stdin)                     |
| `--max-cost-usd=<n>`| no       | Hard budget cap, USD                                   |
| `--max-latency-ms=<n>` | no    | Hard latency cap, milliseconds                         |
| `--json`            | no       | Emit raw `RouteDecision` JSON                          |

Example output:

```
Decision: claude-sonnet-4-6 (anthropic)
  expected cost:    $0.001533
  expected quality: 0.900
  reasoning:        cheapest model meeting quality bar 0.850 for task "qa"; ...

Fallbacks (ordered by next-cheapest):
  1. claude-opus-4-7 (anthropic)   cost-rank 2 of 2 ...

Skipped (4):
  - claude-haiku-4-5: expected quality 0.800 for task "qa" is below quality bar 0.850
  - llama-3.3-70b: expected quality 0.800 for task "qa" is below quality bar 0.850
  - ...
```

### `route frontier --task=<t> [flags]`

Reads `eval/results/frontier.json` and pretty-prints the Pareto frontier
for the requested task. Useful for "what should I pick for codegen at
quality bar 0.85?" Q&A from the terminal.

| Flag                | Required | Description                                                    |
| ------------------- | -------- | -------------------------------------------------------------- |
| `--task=<t>`        | yes      | Task class                                                     |
| `--format=table\|json` | no    | Default `table`                                                |
| `--path=<p>`        | no       | Override the frontier.json path (else uses repo default or `ROUTERLAB_FRONTIER_PATH`) |

### `route models [flags]`

Lists the engine's candidate model pool from
`packages/core/src/candidates.json`.

| Flag                | Description                                |
| ------------------- | ------------------------------------------ |
| `--provider=<p>`    | Filter to one of `anthropic, openai, google, groq, together, hf, openrouter` |
| `--json`            | JSON output                                |

### `route eval frontier --task=<t> [--n=<n>]`

Invokes the frontier builder at `eval/frontier/runner.ts` for a single
task with `n` examples (default per the runner). Persists outcomes to
`eval/results/` and prints a JSON summary on stdout.

The runner is loaded dynamically; override its module path with
`ROUTERLAB_FRONTIER_RUNNER_MODULE` if you've vendored it elsewhere.

### `route version`

Prints `@routerlab/cli` and `@routerlab/core` versions.

## Exit codes

| Code | Meaning                                                 |
| ---- | ------------------------------------------------------- |
| 0    | Success                                                 |
| 1    | No candidates pass the filters                          |
| 2    | Invalid input (bad flag, malformed value, missing arg)  |
| 3    | Downstream error (calibration / runner / missing file)  |

These are stable. Script against them.

## Programmatic use

The CLI is a thin wrapper around `@routerlab/core`. For library access:

```ts
import { route } from "@routerlab/core";

const decision = route({
  task: "qa",
  prompt: "What's the capital of France?",
  qualityBar: 0.85,
  maxCostUsd: 0.005,
});
console.log(decision.chosen.model.model);
```

## Environment variables

| Variable                                | Effect                                                                |
| --------------------------------------- | --------------------------------------------------------------------- |
| `ROUTERLAB_FRONTIER_PATH`               | Override the default `frontier.json` path                             |
| `ROUTERLAB_FRONTIER_RUNNER_MODULE`      | Override the dynamically-imported frontier runner module path         |
| `ROUTERLAB_ATLAS_RESULTS_PATH`          | Atlas calibration file path (read by `@routerlab/core/cost.ts`)       |
| `ROUTERLAB_QUALITY_TABLE_PATH`          | Quality table path (read by `@routerlab/core/quality_predictor.ts`)   |

## License

[Apache-2.0](../../LICENSE)
