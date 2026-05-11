// eval/frontier/build_frontier.test.ts — Pareto computation correctness.
//
// Hand-crafted summaries with known frontiers. No IO; all tests operate
// on in-memory `SummaryRow[]` and assert on the returned `FrontierFile`.

import { describe, expect, test } from "bun:test";
import {
  buildFrontierFile,
  computeParetoFrontier,
} from "./build_frontier.ts";
import type { SummaryRow } from "./_types.ts";

/**
 * Build a minimal `SummaryRow` with sensible defaults. Tests override
 * only the fields they care about.
 */
function row(partial: Partial<SummaryRow> & { task: SummaryRow["task"]; model: string }): SummaryRow {
  return {
    task: partial.task,
    model: partial.model,
    provider: partial.provider ?? "anthropic",
    n: partial.n ?? 10,
    mean_quality: partial.mean_quality ?? 0.5,
    mean_cost_usd: partial.mean_cost_usd ?? 0.001,
    p50_quality: partial.p50_quality ?? 0.5,
    p50_cost_usd: partial.p50_cost_usd ?? 0.001,
    p95_quality: partial.p95_quality ?? 0.7,
    p95_cost_usd: partial.p95_cost_usd ?? 0.002,
    tokenSource: partial.tokenSource ?? "atlas-calibrated",
    confidence: partial.confidence ?? "high",
    errors: partial.errors ?? 0,
  };
}

describe("computeParetoFrontier", () => {
  test("3 models, two-on-frontier, one-dominated", () => {
    // model_a: cheap + low quality   → on frontier
    // model_b: mid cost + mid quality → DOMINATED by model_c (model_c
    //                                   cheaper at lower cost or higher q)
    // Actually carefully craft:
    //   a: cost=0.001, q=0.50  → frontier (cheapest)
    //   b: cost=0.005, q=0.55  → dominated by c? c is cost=0.010, q=0.80
    //                            b is cheaper than c, so b stays on frontier
    //   c: cost=0.010, q=0.80  → frontier (best quality)
    // So with these numbers all 3 are on the frontier. Let's design
    // an explicit dominated row:
    //   d: cost=0.005, q=0.40 → dominated by a (a cheaper and higher q)
    const rows: SummaryRow[] = [
      row({ task: "qa", model: "a", mean_cost_usd: 0.001, mean_quality: 0.5 }),
      row({ task: "qa", model: "b", mean_cost_usd: 0.005, mean_quality: 0.55 }),
      row({ task: "qa", model: "c", mean_cost_usd: 0.01, mean_quality: 0.8 }),
      row({ task: "qa", model: "d", mean_cost_usd: 0.005, mean_quality: 0.4 }),
    ];
    const { frontier, dominated, all } = computeParetoFrontier(rows);
    const frontierIds = frontier.map((f) => f.model).sort();
    expect(frontierIds).toEqual(["a", "b", "c"]);
    expect(dominated.map((d) => d.model)).toEqual(["d"]);
    // `all` sorted by cost-ascending; a (0.001), then b/d at 0.005 (b higher q wins), then c.
    expect(all.map((a) => a.model)).toEqual(["a", "b", "d", "c"]);
    expect(all.find((r) => r.model === "d")?.dominated).toBe(true);
    expect(all.find((r) => r.model === "a")?.dominated).toBe(false);
  });

  test("strict dominance is broken correctly at equal cost", () => {
    // At equal cost, higher quality wins the frontier slot. The lower
    // quality entry at the same cost is dominated.
    const rows: SummaryRow[] = [
      row({ task: "qa", model: "high", mean_cost_usd: 0.005, mean_quality: 0.9 }),
      row({ task: "qa", model: "low", mean_cost_usd: 0.005, mean_quality: 0.3 }),
    ];
    const { frontier, dominated } = computeParetoFrontier(rows);
    expect(frontier.map((f) => f.model)).toEqual(["high"]);
    expect(dominated.map((d) => d.model)).toEqual(["low"]);
  });

  test("strict dominance broken correctly at equal quality", () => {
    // At equal quality, cheaper wins the frontier slot. More-expensive
    // entry at the same quality is dominated.
    const rows: SummaryRow[] = [
      row({ task: "qa", model: "cheap", mean_cost_usd: 0.001, mean_quality: 0.7 }),
      row({ task: "qa", model: "pricey", mean_cost_usd: 0.01, mean_quality: 0.7 }),
    ];
    const { frontier, dominated } = computeParetoFrontier(rows);
    expect(frontier.map((f) => f.model)).toEqual(["cheap"]);
    expect(dominated.map((d) => d.model)).toEqual(["pricey"]);
  });

  test("uninformative rows (n=0) are placed in `all` as dominated and excluded from frontier", () => {
    const rows: SummaryRow[] = [
      row({ task: "qa", model: "good", mean_cost_usd: 0.005, mean_quality: 0.9, n: 50 }),
      row({ task: "qa", model: "errored", mean_cost_usd: 0.001, mean_quality: 0, n: 0 }),
    ];
    const { frontier, all, dominated } = computeParetoFrontier(rows);
    expect(frontier.map((f) => f.model)).toEqual(["good"]);
    // The errored row should be present in `all` and flagged dominated.
    expect(all.map((a) => a.model).sort()).toEqual(["errored", "good"]);
    expect(dominated.map((d) => d.model)).toEqual(["errored"]);
  });

  test("single row: trivially on the frontier", () => {
    const rows: SummaryRow[] = [
      row({ task: "qa", model: "solo", mean_cost_usd: 0.01, mean_quality: 0.6 }),
    ];
    const { frontier, dominated } = computeParetoFrontier(rows);
    expect(frontier.map((f) => f.model)).toEqual(["solo"]);
    expect(dominated.length).toBe(0);
  });

  test("empty input: empty output", () => {
    const { frontier, dominated, all } = computeParetoFrontier([]);
    expect(frontier).toEqual([]);
    expect(dominated).toEqual([]);
    expect(all).toEqual([]);
  });
});

describe("buildFrontierFile", () => {
  test("schema_version is 1; tasks block covers all 5 task classes; provenance distributions are present", () => {
    const rows: SummaryRow[] = [
      row({ task: "qa", model: "a", mean_cost_usd: 0.001, mean_quality: 0.5 }),
      row({ task: "qa", model: "b", mean_cost_usd: 0.005, mean_quality: 0.9 }),
      row({
        task: "codegen",
        model: "c",
        mean_cost_usd: 0.02,
        mean_quality: 0.4,
        tokenSource: "tokenometer-offline",
        confidence: "medium",
      }),
    ];
    const file = buildFrontierFile(rows, { generatedAt: "TEST_TS" });
    expect(file.schema_version).toBe(1);
    expect(file.generated_at).toBe("TEST_TS");
    expect(Object.keys(file.tasks).sort()).toEqual([
      "classification",
      "codegen",
      "qa",
      "reasoning",
      "summarization",
    ]);
    // qa: a and b are both on frontier (a cheaper, b higher q).
    expect(file.tasks.qa.frontier.length).toBe(2);
    expect(file.tasks.qa.dominated.length).toBe(0);
    // codegen: solo entry → frontier of one.
    expect(file.tasks.codegen.frontier.length).toBe(1);
    // Tasks with no rows: empty arrays.
    expect(file.tasks.classification.frontier).toEqual([]);
    expect(file.tasks.summarization.all).toEqual([]);
    expect(file.tasks.reasoning.dominated).toEqual([]);
    // Provenance distribution captures the observed sources.
    expect(file.cost_source.module).toBe("@routerlab/core/cost");
    expect(file.cost_source.token_source_distribution["atlas-calibrated"]).toBe(2);
    expect(file.cost_source.token_source_distribution["tokenometer-offline"]).toBe(1);
    expect(file.cost_source.confidence_distribution.high).toBe(2);
    expect(file.cost_source.confidence_distribution.medium).toBe(1);
  });

  test("atlasResultsPath is recorded when supplied", () => {
    const file = buildFrontierFile([], {
      generatedAt: "TEST_TS",
      atlasResultsPath: "/tmp/atlas/results.json",
    });
    expect(file.cost_source.atlas_results_path).toBe("/tmp/atlas/results.json");
  });

  test("frontier output is deterministic across re-runs given the same input", () => {
    const rows: SummaryRow[] = [
      row({ task: "qa", model: "z", mean_cost_usd: 0.01, mean_quality: 0.5 }),
      row({ task: "qa", model: "a", mean_cost_usd: 0.005, mean_quality: 0.9 }),
      row({ task: "qa", model: "m", mean_cost_usd: 0.001, mean_quality: 0.3 }),
    ];
    const a = buildFrontierFile(rows, { generatedAt: "TS" });
    const b = buildFrontierFile(rows, { generatedAt: "TS" });
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });
});
