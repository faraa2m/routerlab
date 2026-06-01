import { describe, expect, test } from "bun:test";

import {
  BudgetAwareRouter,
  type ModelCandidate,
  type RouteRequest,
} from "../src/index.ts";

const CHEAP: ModelCandidate = {
  provider: "groq",
  model: "llama-3.1-8b",
  pricing: { inputUsdPerMtok: 0.05, outputUsdPerMtok: 0.08 },
  contextWindow: 128000,
};

const EXPENSIVE: ModelCandidate = {
  provider: "anthropic",
  model: "claude-sonnet-4-6",
  pricing: { inputUsdPerMtok: 3, outputUsdPerMtok: 15 },
  contextWindow: 200000,
};

const request = (overrides: Partial<RouteRequest> = {}): RouteRequest => ({
  task: "qa",
  prompt: "Answer the user's support question in one paragraph.",
  qualityBar: 0.5,
  candidates: [CHEAP, EXPENSIVE],
  ...overrides,
});

describe("BudgetAwareRouter", () => {
  test("records actual usage and moves budget state from ok to warn to exhausted", () => {
    const router = new BudgetAwareRouter({ maxBudgetUsd: 0.01, warnAt: 0.5 });

    const first = router.recordActualUsage({
      model: EXPENSIVE,
      usage: { inputTokens: 1000, outputTokens: 100 },
    });
    expect(first.totalUsd).toBeCloseTo(0.0045, 12);
    expect(router.getSnapshot().state).toBe("ok");

    router.recordActualUsage({
      model: EXPENSIVE,
      usage: { inputTokens: 1000, outputTokens: 100 },
    });
    expect(router.getSnapshot().state).toBe("warn");

    router.recordActualUsage({
      model: EXPENSIVE,
      usage: { inputTokens: 1000, outputTokens: 100 },
    });
    const snapshot = router.getSnapshot();
    expect(snapshot.state).toBe("exhausted");
    expect(snapshot.spentUsd).toBeCloseTo(0.0135, 12);
    expect(snapshot.remainingUsd).toBeCloseTo(-0.0035, 12);
  });

  test("caps routeStep by the remaining chain budget", () => {
    const router = new BudgetAwareRouter({ maxBudgetUsd: 0.0001 });
    router.recordActualUsage({
      model: CHEAP,
      usage: { inputTokens: 1000, outputTokens: 0 },
    });

    const routed = router.routeStep(request());

    expect(routed.maxCostUsd).toBeCloseTo(0.00005, 12);
    expect(routed.decision.chosen.expectedCost).toBeLessThanOrEqual(routed.maxCostUsd);
  });

  test("uses degradedQualityBar after warning threshold when configured", () => {
    const router = new BudgetAwareRouter({
      degradedQualityBar: 0.5,
      maxBudgetUsd: 0.01,
      warnAt: 0.5,
    });
    router.recordActualUsage({
      model: EXPENSIVE,
      usage: { inputTokens: 2000, outputTokens: 100 },
    });

    const routed = router.routeStep(request({ qualityBar: 0.9 }));

    expect(routed.effectiveQualityBar).toBe(0.5);
    expect(routed.decision.chosen.model.model).toBe(CHEAP.model);
  });

  test("records an estimated step when actual provider usage is unavailable", () => {
    const router = new BudgetAwareRouter({ maxBudgetUsd: 0.01 });
    const routed = router.routeStep(request());

    const record = router.recordEstimatedStep(routed.decision);
    const snapshot = router.getSnapshot();

    expect(record.source).toBe("estimated");
    expect(record.totalUsd).toBe(routed.decision.chosen.expectedCost);
    expect(snapshot.steps).toHaveLength(1);
  });

  test("throws before routing after the chain budget is exhausted", () => {
    const router = new BudgetAwareRouter({ maxBudgetUsd: 0.001 });
    router.recordActualUsage({
      model: EXPENSIVE,
      usage: { inputTokens: 1000, outputTokens: 0 },
    });

    expect(() => router.routeStep(request())).toThrow(/budget is exhausted/);
  });
});
