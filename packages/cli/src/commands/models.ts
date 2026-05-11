// commands/models.ts — list candidate models from the engine's pool.
//
// Useful for answering "what can routerlab even route to?" without
// digging into `packages/core/src/candidates.json`. Supports an optional
// `--provider=<p>` filter for narrowing the listing.

import { parseArgs } from "node:util";

import { getDefaultCandidates, type ModelCandidate } from "@routerlab/core";

import { CliError, EXIT_INVALID_INPUT } from "../errors.ts";
import { type CliContext, writeLine } from "../io.ts";
import { parseProviderOptional } from "../parse.ts";

/**
 * Parse and execute the `models` subcommand. Prints one model per line in
 * a compact column layout, or JSON when `--json` is set.
 */
export function runModels(ctx: CliContext): number {
  let parsed: ReturnType<typeof parseArgs>;
  try {
    parsed = parseArgs({
      args: [...ctx.argv],
      options: {
        provider: { type: "string" },
        json: { type: "boolean", default: false },
      },
      strict: true,
      allowPositionals: false,
    });
  } catch (e) {
    throw new CliError(
      EXIT_INVALID_INPUT,
      `models: ${e instanceof Error ? e.message : String(e)}`
    );
  }

  const provider = parseProviderOptional(parsed.values["provider"]);
  const json = parsed.values["json"] === true;

  const all = getDefaultCandidates();
  const filtered = provider === undefined ? all : all.filter((c) => c.provider === provider);

  if (json) {
    writeLine(ctx.stdout, JSON.stringify({ candidates: filtered }, null, 2));
    return 0;
  }

  renderTable(filtered, ctx);
  return 0;
}

/**
 * Render the candidate list as a fixed-width table on stdout. Columns are
 * width-padded so values line up; this is a deliberate choice over a CSV /
 * TSV form because the primary audience is humans reading at a terminal.
 */
function renderTable(rows: readonly ModelCandidate[], ctx: CliContext): void {
  if (rows.length === 0) {
    writeLine(ctx.stdout, "(no candidates match filter)");
    return;
  }

  const headers = ["provider", "model", "input $/MTok", "output $/MTok", "context"];
  const data = rows.map((c) => [
    c.provider,
    c.model,
    c.pricing.inputUsdPerMtok.toFixed(2),
    c.pricing.outputUsdPerMtok.toFixed(2),
    c.contextWindow.toLocaleString(),
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
