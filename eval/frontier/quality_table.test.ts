// eval/frontier/quality_table.test.ts — derive predictor-consumable table
// from summary rows. No IO except an optional in-memory test of the disk
// path.

import { describe, expect, test } from "bun:test";
import { buildQualityTable } from "./quality_table.ts";
import type { SummaryRow } from "./_types.ts";

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

describe("buildQualityTable", () => {
  test("trials = n and successes ≈ round(mean * n) for each (model, task)", () => {
    const rows: SummaryRow[] = [
      row({ task: "qa", model: "claude-haiku-4-5", n: 50, mean_quality: 0.84 }),
      row({ task: "codegen", model: "claude-haiku-4-5", n: 30, mean_quality: 0.5 }),
      row({ task: "qa", model: "llama-3.3-70b", n: 50, mean_quality: 0.7 }),
    ];
    const table = buildQualityTable(rows, { generatedAt: "TEST_TS" });
    expect(table.schema_version).toBe(1);
    expect(table.generated_at).toBe("TEST_TS");
    const haiku = table.cells["claude-haiku-4-5"]!;
    expect(haiku.qa).toEqual({ successes: 42, trials: 50 });
    expect(haiku.codegen).toEqual({ successes: 15, trials: 30 });
    const llama = table.cells["llama-3.3-70b"]!;
    expect(llama.qa).toEqual({ successes: 35, trials: 50 });
  });

  test("n=0 rows are omitted (predictor falls back to prior)", () => {
    const rows: SummaryRow[] = [
      row({ task: "qa", model: "errored-model", n: 0, mean_quality: 0 }),
      row({ task: "qa", model: "ok-model", n: 10, mean_quality: 0.6 }),
    ];
    const table = buildQualityTable(rows, { generatedAt: "TS" });
    expect(table.cells["errored-model"]).toBeUndefined();
    expect(table.cells["ok-model"]?.qa).toEqual({ successes: 6, trials: 10 });
  });

  test("clamps mean_quality outside [0, 1] by skipping the cell", () => {
    const rows: SummaryRow[] = [
      row({ task: "qa", model: "bad", n: 10, mean_quality: 1.5 }),
      row({ task: "qa", model: "worse", n: 10, mean_quality: -0.1 }),
      row({ task: "qa", model: "fine", n: 10, mean_quality: 0.5 }),
    ];
    const table = buildQualityTable(rows, { generatedAt: "TS" });
    expect(table.cells["bad"]).toBeUndefined();
    expect(table.cells["worse"]).toBeUndefined();
    expect(table.cells["fine"]?.qa).toEqual({ successes: 5, trials: 10 });
  });

  test("output is byte-stable given the same input", () => {
    const rows: SummaryRow[] = [
      row({ task: "qa", model: "z", n: 5, mean_quality: 0.4 }),
      row({ task: "qa", model: "a", n: 5, mean_quality: 0.8 }),
      row({ task: "codegen", model: "z", n: 5, mean_quality: 0.2 }),
    ];
    const a = JSON.stringify(buildQualityTable(rows, { generatedAt: "TS" }));
    const b = JSON.stringify(buildQualityTable(rows, { generatedAt: "TS" }));
    expect(a).toBe(b);
  });

  test("successes is bounded to [0, trials] even if rounding overshoots", () => {
    // mean = 1.0 exact * trials = 10 → successes = 10. Tests that we
    // can't accidentally end up at 11 successes for 10 trials.
    const rows: SummaryRow[] = [
      row({ task: "qa", model: "perfect", n: 10, mean_quality: 1 }),
    ];
    const table = buildQualityTable(rows, { generatedAt: "TS" });
    expect(table.cells["perfect"]?.qa).toEqual({ successes: 10, trials: 10 });
  });

  test("multiple models for one task; cells keyed by model id", () => {
    const rows: SummaryRow[] = [
      row({ task: "qa", model: "m1", n: 100, mean_quality: 0.5 }),
      row({ task: "qa", model: "m2", n: 100, mean_quality: 0.6 }),
      row({ task: "qa", model: "m3", n: 100, mean_quality: 0.7 }),
    ];
    const table = buildQualityTable(rows, { generatedAt: "TS" });
    expect(Object.keys(table.cells).sort()).toEqual(["m1", "m2", "m3"]);
    expect(table.cells["m1"]?.qa?.successes).toBe(50);
    expect(table.cells["m2"]?.qa?.successes).toBe(60);
    expect(table.cells["m3"]?.qa?.successes).toBe(70);
  });
});
