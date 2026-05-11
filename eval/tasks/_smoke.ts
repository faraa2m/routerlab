#!/usr/bin/env bun
// eval/tasks/_smoke.ts — sanity check that every task loads + parses.
//
// Run: `bun eval/tasks/_smoke.ts`
//
// On first run, fetches small dataset slices from HF datasets-server and
// caches them to `.cache/eval-datasets/`. Subsequent runs hit the cache
// and complete in milliseconds. CI doesn't run this — see the unit tests
// (qa.test.ts, classification.test.ts) which use in-memory fixtures and
// require no network.

import qaTask from "./qa.ts";
import classificationTask from "./classification.ts";
import codegenTask from "./codegen.ts";
import summarizationTask from "./summarization.ts";
import reasoningTask from "./reasoning.ts";
import type { TaskDefinition } from "./_types.ts";

/**
 * Tasks we smoke-test. Tuple shape lets us preserve specific generic
 * params per task without an `any[]` heterogeneous array.
 *
 * Coverage: all five task classes routerlab routes for (qa,
 * classification, codegen, summarization, reasoning). Adding a new task
 * means importing it and dropping it into this list — no other smoke
 * plumbing required.
 */
const tasks: ReadonlyArray<TaskDefinition<unknown, unknown>> = [
  qaTask as TaskDefinition<unknown, unknown>,
  classificationTask as TaskDefinition<unknown, unknown>,
  codegenTask as TaskDefinition<unknown, unknown>,
  summarizationTask as TaskDefinition<unknown, unknown>,
  reasoningTask as TaskDefinition<unknown, unknown>,
];

async function main(): Promise<void> {
  for (const task of tasks) {
    console.log(`\n=== ${task.name} ===`);
    console.log(task.description);

    const examples = await task.loadExamples({ limit: 5, seed: 42 });
    console.log(`loaded ${examples.length} examples`);

    const first = examples[0];
    if (first === undefined) {
      console.log("  (no examples — dataset may be empty)");
      continue;
    }

    const prompt = task.promptTemplate(first.input);
    const promptPreview =
      prompt.length > 280 ? `${prompt.slice(0, 280)}…` : prompt;
    console.log(`id:            ${first.id}`);
    console.log(`reference:     ${JSON.stringify(first.reference)}`);
    console.log(`prompt (≤280): ${promptPreview.replace(/\n/g, " ⏎ ")}`);
  }
  console.log("\nsmoke OK");
}

main().catch((err: unknown) => {
  console.error("smoke failed:", err);
  process.exit(1);
});
