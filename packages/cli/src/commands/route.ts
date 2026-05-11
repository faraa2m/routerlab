// commands/route.ts — the `route` subcommand.
//
// Reads a prompt (from `--input <file>` or stdin), calls the routing engine,
// and prints either a human-friendly table or raw JSON (`--json`).
//
// Exit codes (see ../errors.ts):
//   0 success, 1 no candidates, 2 invalid input, 3 downstream error.
//
// This subcommand is the user-facing surface for routerlab — most users
// will only ever touch this and `frontier`. Output is engineered to be
// scannable: chosen pick first, then ordered fallbacks, then a collapsed
// skipped list with reasons.

import { readFile } from "node:fs/promises";
import type { Writable } from "node:stream";
import { parseArgs } from "node:util";
import { resolve as resolvePath } from "node:path";

import {
  route,
  type RouteDecision,
  type RouteFallback,
  type RoutePick,
  type RouteRequest,
  type RouteSkipped,
} from "@routerlab/core";

import { CliError, EXIT_INVALID_INPUT, EXIT_NO_CANDIDATES } from "../errors.ts";
import { type CliContext, readStdinToString, writeLine } from "../io.ts";
import {
  parseNonNegativeFloatOptional,
  parseQualityBar,
  parseTask,
  type ParsedFlagValue,
} from "../parse.ts";

/**
 * Parse and execute the `route` subcommand against the provided context.
 * Returns the exit code; never throws (all errors are converted).
 */
export async function runRoute(ctx: CliContext): Promise<number> {
  let parsed: ReturnType<typeof parseArgs>;
  try {
    parsed = parseArgs({
      args: [...ctx.argv],
      options: {
        task: { type: "string" },
        "quality-bar": { type: "string" },
        input: { type: "string" },
        "max-cost-usd": { type: "string" },
        "max-latency-ms": { type: "string" },
        json: { type: "boolean", default: false },
      },
      strict: true,
      allowPositionals: false,
    });
  } catch (e) {
    throw new CliError(
      EXIT_INVALID_INPUT,
      `route: ${e instanceof Error ? e.message : String(e)}`
    );
  }

  const task = parseTask(parsed.values["task"]);
  const qualityBar = parseQualityBar(parsed.values["quality-bar"]);
  const maxCostUsd = parseNonNegativeFloatOptional(
    "max-cost-usd",
    parsed.values["max-cost-usd"]
  );
  const maxLatencyMs = parseNonNegativeFloatOptional(
    "max-latency-ms",
    parsed.values["max-latency-ms"]
  );
  const inputPathRaw = parsed.values["input"];
  const json = parsed.values["json"] === true;

  const prompt = await readPrompt(ctx, inputPathRaw);
  if (prompt.trim().length === 0) {
    throw new CliError(
      EXIT_INVALID_INPUT,
      "route: prompt is empty. Pass --input=<file> or pipe text via stdin."
    );
  }

  const request: RouteRequest = {
    task,
    prompt,
    qualityBar,
    ...(maxCostUsd !== undefined ? { maxCostUsd } : {}),
    ...(maxLatencyMs !== undefined ? { maxLatencyMs } : {}),
  };

  let decision: RouteDecision;
  try {
    decision = route(request);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    // The engine throws a Plain `Error` whose message starts with this
    // sentinel when no candidates passed filtering. Map that to exit 1.
    if (msg.includes("no candidates passed filtering")) {
      throw new CliError(EXIT_NO_CANDIDATES, msg);
    }
    // Engine-level validation errors are user input issues.
    if (msg.startsWith("route(): ")) {
      throw new CliError(EXIT_INVALID_INPUT, msg);
    }
    throw e;
  }

  if (json) {
    writeLine(ctx.stdout, JSON.stringify(decision, null, 2));
  } else {
    renderDecision(decision, ctx);
  }
  return 0;
}

/**
 * Resolve the prompt source: `--input <file>` wins over stdin.
 *
 * - If `--input` is given, the file is read from `ctx.cwd`-relative path.
 * - Else stdin is drained to a string (unless the caller is on a TTY, in
 *   which case an empty string is returned and the caller raises a usage
 *   error — see `runRoute` above).
 */
async function readPrompt(
  ctx: CliContext,
  inputPathRaw: ParsedFlagValue
): Promise<string> {
  if (typeof inputPathRaw === "string" && inputPathRaw.length > 0) {
    const absPath = resolvePath(ctx.cwd, inputPathRaw);
    try {
      return await readFile(absPath, "utf8");
    } catch (e) {
      throw new CliError(
        EXIT_INVALID_INPUT,
        `route: cannot read --input "${inputPathRaw}": ${
          e instanceof Error ? e.message : String(e)
        }`
      );
    }
  }
  return await readStdinToString(ctx.stdin);
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/**
 * Render a routing decision as a human-friendly multi-line block to stdout.
 *
 * Layout:
 *
 *   Decision: <model> (<provider>)
 *     expected cost:    $0.001234
 *     expected quality: 0.870
 *     reasoning:        <engine reasoning>
 *
 *   Fallbacks (n):
 *     1. <model> (<provider>)   $cost   q=quality
 *
 *   Skipped (n):
 *     - <model>: <reason>
 */
function renderDecision(decision: RouteDecision, ctx: CliContext): void {
  const out = ctx.stdout;
  renderChosen(decision.chosen, out);

  if (decision.fallbacks.length > 0) {
    writeLine(out, "");
    writeLine(out, "Fallbacks (ordered by next-cheapest):");
    decision.fallbacks.forEach((fb, i) => {
      writeLine(out, formatFallbackLine(fb, i + 1));
    });
  }

  if (decision.skipped.length > 0) {
    writeLine(out, "");
    writeLine(out, `Skipped (${decision.skipped.length}):`);
    for (const s of decision.skipped) {
      writeLine(out, formatSkippedLine(s));
    }
  }
}

function renderChosen(chosen: RoutePick, out: Writable): void {
  writeLine(
    out,
    `Decision: ${chosen.model.model} (${chosen.model.provider})`
  );
  writeLine(out, `  expected cost:    ${formatUsd(chosen.expectedCost)}`);
  writeLine(out, `  expected quality: ${formatQuality(chosen.expectedQuality)}`);
  writeLine(out, `  reasoning:        ${chosen.reasoning}`);
}

function formatFallbackLine(fb: RouteFallback, rank: number): string {
  // The engine encodes cost + quality in the reason string. We display the
  // canonical model/provider line and append the reason for traceability.
  const head = `  ${rank}. ${fb.model.model} (${fb.model.provider})`;
  return `${head}   ${fb.reason}`;
}

function formatSkippedLine(s: RouteSkipped): string {
  return `  - ${s.model.model}: ${s.reason}`;
}

/**
 * Format a USD amount with six fractional digits — that's enough to
 * distinguish models at the low end of the pricing pool (Groq llama-3.1-8b
 * at $0.05/MTok input).
 */
function formatUsd(usd: number): string {
  return `$${usd.toFixed(6)}`;
}

/**
 * Format a quality score to 3 decimal places — matches the engine's own
 * reasoning string format so the two are visually aligned.
 */
function formatQuality(q: number): string {
  return q.toFixed(3);
}
