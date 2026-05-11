// cost.test.ts — behavior tests for `@routerlab/core`'s atlas-grounded
// cost estimator.
//
// The tests cover:
//   - Basic input → positive USD totals and non-zero input tokens.
//   - Caller-provided `expectedOutputTokens` is honored verbatim.
//   - When the atlas calibration file is unavailable, the module falls
//     back gracefully — no exception thrown, `confidence: "medium"`,
//     `tokenSource: "tokenometer-offline"` and a useful note recorded.
//   - When the atlas calibration file IS available (mocked via the
//     `ROUTERLAB_ATLAS_RESULTS_PATH` env var pointing at a tmp JSON),
//     the result switches to `tokenSource: "atlas-calibrated"` and
//     `confidence: "high"`.
//   - The batch API returns one estimate per input.
//   - Invalid input produces a typed `CostEstimationError`.
//   - The Anthropic 1.62× correction factor from tokenometer's prior
//     finding is applied (and visibly inflates the input-token count
//     versus a chars/4 baseline).
//
// No external services. No LLM calls. The cost module is pure; tests
// run in `bun test` and finish in milliseconds.

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  CostEstimationError,
  __resetCalibrationCacheForTest,
  estimateCost,
  estimateCostBatch,
  type CostInput,
} from "../src/index.ts";

const SAMPLE_PROMPT =
  "Summarize the following article in two paragraphs, focusing on the methodology and the headline finding.";

const ANTHROPIC_INPUT: CostInput = {
  prompt: SAMPLE_PROMPT,
  model: "claude-opus-4-7",
  provider: "anthropic",
  pricing: { inputUsdPerMtok: 15, outputUsdPerMtok: 75 },
  taskClass: "qa",
};

const OPENAI_INPUT: CostInput = {
  prompt: SAMPLE_PROMPT,
  model: "gpt-4o",
  provider: "openai",
  pricing: { inputUsdPerMtok: 5, outputUsdPerMtok: 15 },
  taskClass: "qa",
};

const ATLAS_PATH_ENV_VAR = "ROUTERLAB_ATLAS_RESULTS_PATH";

let tmpRoot: string | undefined;
let savedAtlasEnv: string | undefined;

beforeEach(() => {
  savedAtlasEnv = process.env[ATLAS_PATH_ENV_VAR];
  delete process.env[ATLAS_PATH_ENV_VAR];
  __resetCalibrationCacheForTest();
});

afterEach(() => {
  if (savedAtlasEnv === undefined) {
    delete process.env[ATLAS_PATH_ENV_VAR];
  } else {
    process.env[ATLAS_PATH_ENV_VAR] = savedAtlasEnv;
  }
  if (tmpRoot !== undefined) {
    rmSync(tmpRoot, { recursive: true, force: true });
    tmpRoot = undefined;
  }
  __resetCalibrationCacheForTest();
});

describe("estimateCost — basic shape", () => {
  test("returns positive USD totals and non-zero input tokens for a normal prompt", () => {
    // Force atlas path to a non-existent location so we exercise the
    // fallback path deterministically.
    process.env[ATLAS_PATH_ENV_VAR] = "/nonexistent/path/results.json";

    const result = estimateCost(ANTHROPIC_INPUT);

    expect(result.model).toBe("claude-opus-4-7");
    expect(result.provider).toBe("anthropic");
    expect(result.inputTokens).toBeGreaterThan(0);
    expect(result.outputTokensEstimate).toBe(100); // QA default
    expect(result.inputUsd).toBeGreaterThan(0);
    expect(result.outputUsd).toBeGreaterThan(0);
    expect(result.totalUsd).toBe(result.inputUsd + result.outputUsd);
    expect(result.notes.length).toBeGreaterThan(0);
  });

  test("Anthropic input has the ~1.62× fallback correction visibly applied", () => {
    // Atlas missing → fallback table applies. The fallback Anthropic
    // correction is 1.62×, so the calibrated input-token count should be
    // strictly larger than the raw cl100k_base count.
    process.env[ATLAS_PATH_ENV_VAR] = "/nonexistent/path/results.json";

    const result = estimateCost(ANTHROPIC_INPUT);

    // SAMPLE_PROMPT cl100k_base raw count is in the ~17–22 token range.
    // After 1.62× it should land in the ~27–36 range. Use generous bounds
    // so the assertion remains stable across tokenometer dep bumps.
    expect(result.inputTokens).toBeGreaterThan(20);
    expect(result.inputTokens).toBeLessThan(100);
    expect(result.tokenSource).toBe("tokenometer-offline");
    expect(result.confidence).toBe("medium");
    const calibrationNote = result.notes.find((n) =>
      n.includes("fallback calibration applied: factor 1.620"),
    );
    expect(calibrationNote).toBeDefined();
  });
});

describe("estimateCost — output-token handling", () => {
  test("honors caller-provided expectedOutputTokens exactly", () => {
    process.env[ATLAS_PATH_ENV_VAR] = "/nonexistent/path/results.json";

    const result = estimateCost({
      ...ANTHROPIC_INPUT,
      expectedOutputTokens: 42,
    });

    expect(result.outputTokensEstimate).toBe(42);
    expect(result.outputUsd).toBeCloseTo(
      (42 / 1_000_000) * ANTHROPIC_INPUT.pricing.outputUsdPerMtok,
      12,
    );
  });

  test("defaults output tokens by task class when no caller override", () => {
    process.env[ATLAS_PATH_ENV_VAR] = "/nonexistent/path/results.json";

    const expectedByTaskClass: ReadonlyArray<readonly [CostInput["taskClass"], number]> = [
      ["classification", 10],
      ["qa", 100],
      ["summarization", 200],
      ["codegen", 300],
      ["reasoning", 500],
    ];

    for (const [taskClass, expected] of expectedByTaskClass) {
      const result = estimateCost({ ...OPENAI_INPUT, taskClass });
      expect(result.outputTokensEstimate).toBe(expected);
    }
  });
});

describe("estimateCost — calibration fallback graceful degradation", () => {
  test("when atlas file is missing, falls back without throwing", () => {
    process.env[ATLAS_PATH_ENV_VAR] = "/this/path/does/not/exist.json";

    let result;
    expect(() => {
      result = estimateCost(ANTHROPIC_INPUT);
    }).not.toThrow();
    if (result === undefined) throw new Error("result should be set");
    expect(result.tokenSource).toBe("tokenometer-offline");
    expect(result.confidence).toBe("medium");
  });

  test("when atlas file is malformed (not JSON), throws typed CostEstimationError", () => {
    tmpRoot = mkdtempSync(join(tmpdir(), "routerlab-cost-test-"));
    const malformed = join(tmpRoot, "results.json");
    writeFileSync(malformed, "this-is-not-json", "utf8");
    process.env[ATLAS_PATH_ENV_VAR] = malformed;

    try {
      estimateCost(ANTHROPIC_INPUT);
      throw new Error("estimateCost should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(CostEstimationError);
      if (err instanceof CostEstimationError) {
        expect(err.failure.kind).toBe("calibration-malformed");
      }
    }
  });
});

describe("estimateCost — atlas-calibrated path", () => {
  test("when valid atlas results.json is available, reports atlas-calibrated/high confidence", () => {
    tmpRoot = mkdtempSync(join(tmpdir(), "routerlab-cost-test-"));
    const atlasPath = join(tmpRoot, "results.json");
    const atlasPayload = {
      schema_version: 1,
      generated_at: "2026-05-10T00:00:00Z",
      correction_factors: {
        anthropic: {
          cl100k_base: { median: 1.62, p25: 1.55, p75: 1.68, sample_size: 8742 },
          default: { median: 1.62, p25: 1.55, p75: 1.68, sample_size: 8742 },
        },
        openai: {
          o200k_base: { median: 1.0, p25: 1.0, p75: 1.0, sample_size: 8742 },
        },
      },
    };
    writeFileSync(atlasPath, JSON.stringify(atlasPayload), "utf8");
    process.env[ATLAS_PATH_ENV_VAR] = atlasPath;

    const result = estimateCost(ANTHROPIC_INPUT);

    expect(result.tokenSource).toBe("atlas-calibrated");
    expect(result.confidence).toBe("high");
    const atlasNote = result.notes.find((n) => n.includes("atlas calibration applied"));
    expect(atlasNote).toBeDefined();
    if (atlasNote === undefined) throw new Error("unreachable");
    expect(atlasNote).toContain("anthropic/cl100k_base");
    expect(atlasNote).toContain("generated_at=2026-05-10T00:00:00Z");
    expect(atlasNote).toContain("n=8742");
  });

  test("openai with factor=1.0 in atlas is reported as offline, not atlas-calibrated", () => {
    // A 1.0 factor is not a *correction*; it should not be advertised as
    // calibration applied. This makes the high-confidence claim honest:
    // we only stamp "atlas-calibrated/high" when the factor actually
    // moves the count.
    tmpRoot = mkdtempSync(join(tmpdir(), "routerlab-cost-test-"));
    const atlasPath = join(tmpRoot, "results.json");
    writeFileSync(
      atlasPath,
      JSON.stringify({
        schema_version: 1,
        generated_at: "2026-05-10T00:00:00Z",
        correction_factors: {
          openai: { o200k_base: { median: 1.0, sample_size: 1000 } },
        },
      }),
      "utf8",
    );
    process.env[ATLAS_PATH_ENV_VAR] = atlasPath;

    const result = estimateCost(OPENAI_INPUT);

    expect(result.tokenSource).toBe("tokenometer-offline");
    expect(result.confidence).toBe("medium");
  });
});

describe("estimateCostBatch", () => {
  test("returns one estimate per input, in order, and is equivalent to .map(estimateCost)", () => {
    process.env[ATLAS_PATH_ENV_VAR] = "/nonexistent/path/results.json";

    const inputs: CostInput[] = [
      ANTHROPIC_INPUT,
      { ...OPENAI_INPUT, taskClass: "codegen" },
      {
        prompt: "Classify the sentiment of: I love this.",
        model: "llama-3.1-8b",
        provider: "groq",
        pricing: { inputUsdPerMtok: 0.05, outputUsdPerMtok: 0.08 },
        taskClass: "classification",
      },
    ];

    const batch = estimateCostBatch(inputs);
    expect(batch).toHaveLength(3);
    expect(batch[0]?.model).toBe("claude-opus-4-7");
    expect(batch[1]?.model).toBe("gpt-4o");
    expect(batch[2]?.model).toBe("llama-3.1-8b");
    expect(batch[2]?.outputTokensEstimate).toBe(10); // classification default

    // Reset cache so the .map() path uses the same calibration state.
    __resetCalibrationCacheForTest();
    const oneByOne = inputs.map(estimateCost);
    for (let i = 0; i < batch.length; i++) {
      expect(batch[i]?.inputTokens).toBe(oneByOne[i]?.inputTokens);
      expect(batch[i]?.totalUsd).toBe(oneByOne[i]?.totalUsd);
      expect(batch[i]?.tokenSource).toBe(oneByOne[i]?.tokenSource);
    }
  });
});

describe("estimateCost — purity & idempotence", () => {
  test("same input → same output across repeated calls", () => {
    process.env[ATLAS_PATH_ENV_VAR] = "/nonexistent/path/results.json";

    const a = estimateCost(ANTHROPIC_INPUT);
    const b = estimateCost(ANTHROPIC_INPUT);
    expect(b.inputTokens).toBe(a.inputTokens);
    expect(b.totalUsd).toBe(a.totalUsd);
    expect(b.tokenSource).toBe(a.tokenSource);
    expect(b.confidence).toBe(a.confidence);
  });

  test("does not mutate the input object", () => {
    process.env[ATLAS_PATH_ENV_VAR] = "/nonexistent/path/results.json";

    const input: CostInput = { ...ANTHROPIC_INPUT };
    const snapshot = JSON.stringify(input);
    estimateCost(input);
    expect(JSON.stringify(input)).toBe(snapshot);
  });
});

describe("estimateCost — input validation", () => {
  test("unknown provider throws typed CostEstimationError", () => {
    process.env[ATLAS_PATH_ENV_VAR] = "/nonexistent/path/results.json";

    try {
      // Force a bad provider via cast — runtime check should catch it.
      estimateCost({
        ...ANTHROPIC_INPUT,
        provider: "made-up" as unknown as CostInput["provider"],
      });
      throw new Error("estimateCost should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(CostEstimationError);
      if (err instanceof CostEstimationError) {
        expect(err.failure.kind).toBe("unknown-provider");
      }
    }
  });

  test("negative pricing throws typed CostEstimationError", () => {
    process.env[ATLAS_PATH_ENV_VAR] = "/nonexistent/path/results.json";

    try {
      estimateCost({
        ...ANTHROPIC_INPUT,
        pricing: { inputUsdPerMtok: -1, outputUsdPerMtok: 75 },
      });
      throw new Error("estimateCost should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(CostEstimationError);
      if (err instanceof CostEstimationError) {
        expect(err.failure.kind).toBe("invalid-input");
      }
    }
  });

  test("non-string prompt throws typed CostEstimationError", () => {
    process.env[ATLAS_PATH_ENV_VAR] = "/nonexistent/path/results.json";

    try {
      estimateCost({
        ...ANTHROPIC_INPUT,
        prompt: 123 as unknown as string,
      });
      throw new Error("estimateCost should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(CostEstimationError);
      if (err instanceof CostEstimationError) {
        expect(err.failure.kind).toBe("invalid-input");
      }
    }
  });
});

describe("estimateCost — hosting-platform provider mapping", () => {
  test("groq + llama routes through openai cl100k/o200k path with a note", () => {
    process.env[ATLAS_PATH_ENV_VAR] = "/nonexistent/path/results.json";

    const result = estimateCost({
      prompt: "Hello, world! This is a llama prompt.",
      model: "llama-3.3-70b",
      provider: "groq",
      pricing: { inputUsdPerMtok: 0.59, outputUsdPerMtok: 0.79 },
      taskClass: "qa",
    });

    expect(result.inputTokens).toBeGreaterThan(0);
    const mappingNote = result.notes.find((n) =>
      n.includes("mapped routerlab provider \"groq\""),
    );
    expect(mappingNote).toBeDefined();
  });

  test("together + mixtral routes through mistral SentencePiece path", () => {
    process.env[ATLAS_PATH_ENV_VAR] = "/nonexistent/path/results.json";

    const result = estimateCost({
      prompt: "Hello, mixtral.",
      model: "mixtral-8x7b",
      provider: "together",
      pricing: { inputUsdPerMtok: 0.24, outputUsdPerMtok: 0.24 },
      taskClass: "qa",
    });

    expect(result.inputTokens).toBeGreaterThan(0);
    const mappingNote = result.notes.find((n) =>
      n.includes("mapped routerlab provider \"together\"") && n.includes("\"mistral\""),
    );
    expect(mappingNote).toBeDefined();
  });
});
