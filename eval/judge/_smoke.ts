#!/usr/bin/env bun
// eval/judge/_smoke.ts — single live judge call as a connectivity check.
//
// Run with `bun eval/judge/_smoke.ts`. Behavior:
//   - If `ANTHROPIC_API_KEY` is missing: skip (exit 0).
//   - Else: issue one judge call against a hand-crafted summarization
//     fixture and print the parsed score + reasoning preview.
//
// Exit codes:
//   - 0 on success or skip.
//   - 1 on failure.

import { judge } from "./judge.ts";
import type { JudgeRequest } from "./_types.ts";

const FIXTURE: JudgeRequest = {
  taskClass: "summarization",
  prompt:
    "Summarize the following article in a single sentence.\n\nArticle: The James Webb Space Telescope detected water vapor in the atmosphere of a small rocky exoplanet for the first time, scientists announced Tuesday.\n\nSummary:",
  candidate:
    "Scientists used the James Webb Space Telescope to detect water vapor on a rocky exoplanet for the first time.",
  reference:
    "JWST detected water vapor in the atmosphere of a small rocky exoplanet, a first for the field.",
};

async function main(): Promise<void> {
  const haveKey = Boolean(process.env["ANTHROPIC_API_KEY"]);
  if (!haveKey) {
    console.log("[judge:smoke] SKIP (no ANTHROPIC_API_KEY in env)");
    process.exit(0);
  }
  console.log("[judge:smoke] running one live judge call with default model");
  try {
    const resp = await judge(FIXTURE);
    const reasoningPreview = resp.reasoning.slice(0, 120).replace(/\s+/g, " ");
    console.log(
      `[judge:smoke] OK   model=${resp.judge_model} score=${resp.score.toFixed(3)} ` +
        `cacheHit=${resp.cacheHit} reasoning="${reasoningPreview}…"`,
    );
    process.exit(0);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error(`[judge:smoke] FAIL ${msg}`);
    process.exit(1);
  }
}

const isMain = (() => {
  try {
    const meta = import.meta as ImportMeta & { main?: boolean };
    if (typeof meta.main === "boolean") return meta.main;
  } catch {
    /* fall through */
  }
  return process.argv[1]?.endsWith("_smoke.ts") ?? false;
})();

if (isMain) {
  await main();
}
