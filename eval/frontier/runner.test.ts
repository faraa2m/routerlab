// eval/frontier/runner.test.ts — smoke run with mocked runners.
//
// Tests the build orchestrator end-to-end without any network calls. The runner
// factory is mocked to return deterministic stubs whose outputs survive
// each task's `parseOutput`/`score` path.

import { describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, existsSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  aggregateBucket,
  outcomePath,
  promptDigest,
  quantile,
  runFrontier,
  runOne,
  TASKS,
  type RunnerFactory,
} from "./runner.ts";
import type { RunOutcome, SummaryRow } from "./_types.ts";
import type { ModelCandidate } from "../../packages/core/src/types.ts";
import type { Runner } from "../runners/_types.ts";

// ---------------------------------------------------------------------------
// Test scaffolding
// ---------------------------------------------------------------------------

/** Stub candidate list: 2 models across 2 providers. */
const TEST_CANDIDATES: ModelCandidate[] = [
  {
    provider: "anthropic",
    model: "claude-haiku-4-5",
    pricing: { inputUsdPerMtok: 1, outputUsdPerMtok: 5 },
    contextWindow: 200_000,
  },
  {
    provider: "groq",
    model: "llama-3.1-8b",
    pricing: { inputUsdPerMtok: 0.05, outputUsdPerMtok: 0.08 },
    contextWindow: 128_000,
  },
];

/**
 * Returns a stub runner. The output is driven by a fixed template per
 * task — when the prompt looks like a classification prompt we return
 * "neutral", a reasoning prompt we return "#### 42", and so on. Every
 * other prompt returns a generic answer. Token counts and latency are
 * fixed for determinism.
 */
function makeStubRunner(modelToOutput?: Map<string, string>): Runner {
  return {
    provider: "anthropic",
    listModels: () => ["claude-haiku-4-5", "llama-3.1-8b"],
    async run(req): Promise<{
      model: string;
      output: string;
      inputTokens?: number;
      outputTokens?: number;
      usdCost?: number;
      latencyMs: number;
      ts: string;
    }> {
      const overrides = modelToOutput ?? new Map<string, string>();
      const lower = req.prompt.toLowerCase();
      let out = overrides.get(req.model);
      if (out === undefined) {
        if (lower.includes("classify this tweet")) out = "neutral";
        else if (lower.includes("solve this problem step by step")) out = "#### 0";
        else if (lower.includes("summarize the following article")) out = "A short summary.";
        else if (lower.includes("complete the following python")) out = "    return 0\n";
        else out = "fixture answer";
      }
      return {
        model: req.model,
        output: out,
        inputTokens: 16,
        outputTokens: 8,
        usdCost: 0,
        latencyMs: 1,
        ts: "1970-01-01T00:00:00.000Z",
      };
    },
  };
}

const stubRunnerFactory: RunnerFactory = () => makeStubRunner();

function makeTmpResults(): { dir: string; cleanup: () => void } {
  const dir = mkdtempSync(join(tmpdir(), "routerlab-frontier-"));
  return {
    dir,
    cleanup: () => {
      rmSync(dir, { recursive: true, force: true });
    },
  };
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

describe("quantile", () => {
  test("p50 of an even list returns the linear-interp midpoint", () => {
    expect(quantile([1, 2, 3, 4], 0.5)).toBe(2.5);
  });
  test("p95 of [0..9] is 8.55 (linear interp)", () => {
    expect(quantile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 0.95)).toBeCloseTo(8.55, 5);
  });
  test("empty list returns 0", () => {
    expect(quantile([], 0.5)).toBe(0);
  });
});

describe("promptDigest", () => {
  test("digest is deterministic and 16 hex chars", () => {
    const a = promptDigest("hello");
    const b = promptDigest("hello");
    expect(a).toBe(b);
    expect(a.length).toBe(16);
    expect(a).toMatch(/^[a-f0-9]{16}$/);
  });
});

describe("aggregateBucket", () => {
  test("mean / p50 / p95 / token-source modal value all computed from outcomes", () => {
    const outcomes: RunOutcome[] = [
      {
        task: "qa",
        model: "m1",
        provider: "anthropic",
        exampleId: "e1",
        promptHash: "deadbeef00000000",
        output: "x",
        qualityScore: 0.2,
        scoreSource: "task-score",
        costUsd: 0.001,
        tokenSource: "atlas-calibrated",
        confidence: "high",
        latencyMs: 1,
        ts: "1970-01-01T00:00:00.000Z",
      },
      {
        task: "qa",
        model: "m1",
        provider: "anthropic",
        exampleId: "e2",
        promptHash: "deadbeef00000001",
        output: "y",
        qualityScore: 0.8,
        scoreSource: "task-score",
        costUsd: 0.002,
        tokenSource: "atlas-calibrated",
        confidence: "high",
        latencyMs: 2,
        ts: "1970-01-01T00:00:01.000Z",
      },
      {
        task: "qa",
        model: "m1",
        provider: "anthropic",
        exampleId: "e3",
        promptHash: "deadbeef00000002",
        output: "z",
        qualityScore: 0.5,
        scoreSource: "task-score",
        costUsd: 0.003,
        tokenSource: "atlas-calibrated",
        confidence: "high",
        latencyMs: 3,
        ts: "1970-01-01T00:00:02.000Z",
      },
    ];
    const row = aggregateBucket("qa", outcomes);
    expect(row.n).toBe(3);
    expect(row.errors).toBe(0);
    expect(row.mean_quality).toBeCloseTo((0.2 + 0.5 + 0.8) / 3, 6);
    expect(row.p50_quality).toBeCloseTo(0.5, 6);
    expect(row.tokenSource).toBe("atlas-calibrated");
    expect(row.confidence).toBe("high");
    expect(row.tokenSourceNote).toBeUndefined();
  });

  test("errored outcomes are counted in `errors` and excluded from aggregates", () => {
    const outcomes: RunOutcome[] = [
      {
        task: "qa",
        model: "m1",
        provider: "anthropic",
        exampleId: "e1",
        promptHash: "x",
        output: "",
        qualityScore: 0,
        scoreSource: "task-score",
        costUsd: 0.001,
        tokenSource: "tokenometer-offline",
        confidence: "medium",
        latencyMs: 0,
        ts: "1970-01-01T00:00:00.000Z",
        error: "boom",
      },
      {
        task: "qa",
        model: "m1",
        provider: "anthropic",
        exampleId: "e2",
        promptHash: "y",
        output: "ok",
        qualityScore: 0.9,
        scoreSource: "task-score",
        costUsd: 0.002,
        tokenSource: "tokenometer-offline",
        confidence: "medium",
        latencyMs: 1,
        ts: "1970-01-01T00:00:01.000Z",
      },
    ];
    const row = aggregateBucket("qa", outcomes);
    expect(row.n).toBe(1);
    expect(row.errors).toBe(1);
    expect(row.mean_quality).toBe(0.9);
    expect(row.mean_cost_usd).toBe(0.002);
  });

  test("disagreeing token-sources record a note", () => {
    const base: Omit<RunOutcome, "tokenSource" | "exampleId" | "promptHash"> = {
      task: "qa",
      model: "m1",
      provider: "anthropic",
      output: "x",
      qualityScore: 0.5,
      scoreSource: "task-score",
      costUsd: 0.001,
      confidence: "high",
      latencyMs: 1,
      ts: "1970-01-01T00:00:00.000Z",
    };
    const outcomes: RunOutcome[] = [
      { ...base, exampleId: "e1", promptHash: "a", tokenSource: "atlas-calibrated" },
      { ...base, exampleId: "e2", promptHash: "b", tokenSource: "atlas-calibrated" },
      { ...base, exampleId: "e3", promptHash: "c", tokenSource: "tokenometer-offline" },
    ];
    const row = aggregateBucket("qa", outcomes);
    expect(row.tokenSource).toBe("atlas-calibrated");
    expect(row.tokenSourceNote).toContain("mixed");
  });
});

// ---------------------------------------------------------------------------
// End-to-end smoke
// ---------------------------------------------------------------------------

describe("runFrontier (mocked runners)", () => {
  test("smoke run produces summary rows + persisted outcomes", async () => {
    const { dir, cleanup } = makeTmpResults();
    try {
      const { summary, outcomes } = await runFrontier({
        tasks: ["qa", "classification"],
        candidates: TEST_CANDIDATES,
        examplesPerTask: 1,
        seed: 7,
        runnerFactory: stubRunnerFactory,
        resultsDir: dir,
        quiet: true,
      });
      // 2 tasks * 2 candidates = 4 summary rows.
      expect(summary.length).toBe(4);
      // 4 buckets * 1 example each = 4 outcomes.
      expect(outcomes.length).toBe(4);
      for (const o of outcomes) {
        expect(o.error).toBeUndefined();
        // Cost is atlas-grounded via cost.ts even when the runner stub
        // reports usdCost=0 — the build orchestrator never trusts runner-side
        // cost. This is the load-bearing audit property: every row's
        // cost has tokenSource provenance from cost.ts.
        expect(o.costUsd).toBeGreaterThanOrEqual(0);
        expect(o.tokenSource).toMatch(
          /^(tokenometer-empirical|tokenometer-offline|atlas-calibrated|proxy)$/,
        );
      }
      // summary.json persisted.
      expect(existsSync(join(dir, "summary.json"))).toBe(true);
      const raw = readFileSync(join(dir, "summary.json"), "utf8");
      const parsed = JSON.parse(raw) as { rows: SummaryRow[] };
      expect(parsed.rows.length).toBe(4);
      // each outcome persisted at its derived path
      for (const o of outcomes) {
        const p = outcomePath(dir, o);
        expect(existsSync(p)).toBe(true);
      }
    } finally {
      cleanup();
    }
  });

  test("missing runner factory does not crash; skipped provider produces errored outcomes", async () => {
    const { dir, cleanup } = makeTmpResults();
    try {
      // factory throws for groq, succeeds for anthropic.
      const factory: RunnerFactory = (p) => {
        if (p === "groq") throw new Error("missing GROQ_API_KEY");
        return makeStubRunner();
      };
      const { summary } = await runFrontier({
        tasks: ["qa"],
        candidates: TEST_CANDIDATES,
        examplesPerTask: 1,
        runnerFactory: factory,
        resultsDir: dir,
        quiet: true,
      });
      const haiku = summary.find((r) => r.model === "claude-haiku-4-5")!;
      const llama = summary.find((r) => r.model === "llama-3.1-8b")!;
      expect(haiku.n).toBe(1);
      expect(haiku.errors).toBe(0);
      expect(llama.n).toBe(0);
      expect(llama.errors).toBe(1);
    } finally {
      cleanup();
    }
  });

  test("resume=true reuses existing on-disk outcomes for already-complete buckets", async () => {
    const { dir, cleanup } = makeTmpResults();
    try {
      // First pass: populate the bucket.
      const r1 = await runFrontier({
        tasks: ["qa"],
        candidates: [TEST_CANDIDATES[0]!],
        examplesPerTask: 1,
        runnerFactory: stubRunnerFactory,
        resultsDir: dir,
        quiet: true,
      });
      expect(r1.summary[0]?.n).toBe(1);
      // Second pass with resume=true: should not call the runner again.
      // We use a sentinel that throws to verify.
      const throwingFactory: RunnerFactory = () => ({
        provider: "anthropic",
        listModels: () => [],
        async run(): Promise<never> {
          throw new Error("should not be called when resuming");
        },
      });
      const r2 = await runFrontier({
        tasks: ["qa"],
        candidates: [TEST_CANDIDATES[0]!],
        examplesPerTask: 1,
        runnerFactory: throwingFactory,
        resultsDir: dir,
        quiet: true,
        resume: true,
      });
      expect(r2.summary[0]?.n).toBe(1);
    } finally {
      cleanup();
    }
  });
});

// ---------------------------------------------------------------------------
// Targeted runOne path: judge wins over task-score when supplied
// ---------------------------------------------------------------------------

describe("runOne with judge override", () => {
  test("judge score overrides task-score; scoreSource records 'judge'", async () => {
    // Use a fake qa example whose reference would yield 0 via task-score,
    // but the judge returns 0.9.
    const example = {
      id: "e",
      input: { context: "x", question: "y" },
      reference: { goldAnswers: ["zzz"], isImpossible: false },
    };
    const candidate = TEST_CANDIDATES[0]!;
    const runner = makeStubRunner();
    const outcome = await runOne({
      taskClass: "qa",
      candidate,
      example,
      runner,
      judge: async () => ({ score: 0.9 }),
    });
    expect(outcome.qualityScore).toBe(0.9);
    expect(outcome.scoreSource).toBe("judge");
  });

  test("judge scores outside [0, 1] are clamped", async () => {
    const example = {
      id: "e",
      input: { context: "x", question: "y" },
      reference: { goldAnswers: ["zzz"], isImpossible: false },
    };
    const candidate = TEST_CANDIDATES[0]!;
    const runner = makeStubRunner();
    const outcome = await runOne({
      taskClass: "qa",
      candidate,
      example,
      runner,
      judge: async () => ({ score: 1.7 }),
    });
    expect(outcome.qualityScore).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Anchor: TASKS contract sanity — must cover all 5 classes the brief lists
// ---------------------------------------------------------------------------

describe("TASKS registry", () => {
  test("covers exactly the 5 canonical task classes", () => {
    expect(Object.keys(TASKS).sort()).toEqual([
      "classification",
      "codegen",
      "qa",
      "reasoning",
      "summarization",
    ]);
  });
});
