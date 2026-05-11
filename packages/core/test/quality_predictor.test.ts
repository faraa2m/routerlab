// quality_predictor.test.ts — behavior tests for the calibrated
// quality predictor.
//
// Covered cases:
//   - File missing → falls back to the seeded prior in `quality_prior.ts`.
//   - File present → returns the measured mean and a Wilson 95% CI
//     computed from the on-disk (successes, trials) counts.
//   - `wilsonScore95`: hand-computed reference for (10, 10) and (5, 10).
//   - Memoization: repeat calls without resetting the cache yield the
//     same result even if the underlying file changes; reset re-reads.
//   - Unknown (taskClass, modelId) returns the uniform-prior fallback
//     with `n = PRIOR_N`.
//   - Schema-tolerance: a malformed file with no usable cells silently
//     falls back to the prior (mirrors cost.ts's tolerance contract).
//
// No external services. No LLM calls. Pure-TS unit tests, run via `bun test`.

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  __resetQualityCacheForTest,
  getQualitySourceInfo,
  PRIOR_N,
  predictQuality,
  predictQualityWithCI,
  wilsonScore95,
  type QualityWithCI,
} from "../src/index.ts";
import { predictQuality as priorPredictQuality } from "../src/quality_prior.ts";

const QUALITY_TABLE_PATH_ENV_VAR = "ROUTERLAB_QUALITY_TABLE_PATH";

let tmpRoot: string | undefined;
let savedEnv: string | undefined;

beforeEach(() => {
  savedEnv = process.env[QUALITY_TABLE_PATH_ENV_VAR];
  // Force the predictor to look at a non-existent file so the default
  // path on disk (eval/results/quality_table.json, which may or may not
  // exist depending on harness state) does not influence test outcomes.
  process.env[QUALITY_TABLE_PATH_ENV_VAR] = "/nonexistent/quality_table.json";
  __resetQualityCacheForTest();
});

afterEach(() => {
  if (savedEnv === undefined) {
    delete process.env[QUALITY_TABLE_PATH_ENV_VAR];
  } else {
    process.env[QUALITY_TABLE_PATH_ENV_VAR] = savedEnv;
  }
  if (tmpRoot !== undefined) {
    rmSync(tmpRoot, { recursive: true, force: true });
    tmpRoot = undefined;
  }
  __resetQualityCacheForTest();
});

// ---------------------------------------------------------------------------
// wilsonScore95 — pure math reference checks
// ---------------------------------------------------------------------------

describe("wilsonScore95", () => {
  test("(10 successes, 10 trials) matches hand-computed reference", () => {
    // Hand-computed at the 95% level using z = 1.959963984540054:
    //   p = 1.0, n = 10, z^2 = 3.841459
    //   denom    = 1 + 0.384146  = 1.384146
    //   center   = (1.0 + 0.192073) / 1.384146 ≈ 0.86123
    //   radius   = z * sqrt(0 + 0.009604) / 1.384146 ≈ 0.13877
    //   (lo, hi) ≈ (0.72246, 1.00000)
    const { lo, hi } = wilsonScore95(10, 10);
    expect(lo).toBeGreaterThan(0.72);
    expect(lo).toBeLessThan(0.73);
    expect(hi).toBe(1); // clamped at the upper bound
  });

  test("(5 successes, 10 trials) is symmetric around 0.5", () => {
    // p = 0.5, n = 10. Wilson interval is symmetric around the adjusted
    // center, which itself sits at 0.5 when p = 0.5. The closed-form value:
    //   center = 0.5, radius = z * sqrt(0.025 + 0.009604) / 1.384146
    //          ≈ 1.959964 * 0.18603 / 1.384146 ≈ 0.26340
    //   (lo, hi) ≈ (0.23659, 0.76340)
    const { lo, hi } = wilsonScore95(5, 10);
    expect(lo).toBeGreaterThan(0.23);
    expect(lo).toBeLessThan(0.24);
    expect(hi).toBeGreaterThan(0.76);
    expect(hi).toBeLessThan(0.77);
    // Symmetry around 0.5
    expect(Math.abs(0.5 - lo - (hi - 0.5))).toBeLessThan(1e-9);
  });

  test("(0 successes, 10 trials) is well-defined and lo is exactly 0", () => {
    // Wilson does not collapse at the boundaries (Wald does). Lo must be
    // clamped to 0 from the analytical lower bound, hi must be positive.
    const { lo, hi } = wilsonScore95(0, 10);
    expect(lo).toBe(0);
    expect(hi).toBeGreaterThan(0);
    expect(hi).toBeLessThan(1);
  });

  test("rejects invalid arguments", () => {
    expect(() => wilsonScore95(0, 0)).toThrow(/positive integer/);
    expect(() => wilsonScore95(-1, 10)).toThrow(/in \[0, trials\]/);
    expect(() => wilsonScore95(11, 10)).toThrow(/in \[0, trials\]/);
    expect(() => wilsonScore95(Number.NaN, 10)).toThrow(/finite/);
    expect(() => wilsonScore95(5, Number.NaN)).toThrow(/finite/);
  });

  test("interval narrows as n grows for the same proportion", () => {
    const small = wilsonScore95(8, 10);
    const big = wilsonScore95(800, 1000);
    const smallWidth = small.hi - small.lo;
    const bigWidth = big.hi - big.lo;
    expect(bigWidth).toBeLessThan(smallWidth);
  });
});

// ---------------------------------------------------------------------------
// Fallback path: no measured file present
// ---------------------------------------------------------------------------

describe("predictQuality* — fallback to seeded prior", () => {
  test("predictQuality stays within one quantization step of quality_prior", () => {
    // The predictor synthesizes (successes, trials) from the prior
    // via `round(p * PRIOR_N)` capped to `[1, PRIOR_N - 1]`. That clamp
    // is intentional (priors are weak evidence; see buildPriorTable
    // docstring). For values strictly inside (0.1, 0.9) the predictor
    // reproduces the prior within ±1/(2*PRIOR_N); near the boundaries
    // the cap can pull the value in by up to 1/PRIOR_N. Use the looser
    // tolerance for the boundary case.
    const tick = 1 / PRIOR_N;
    const cases: Array<{ task: "qa" | "codegen" | "reasoning" | "classification" | "summarization"; model: string }> = [
      { task: "qa", model: "claude-opus-4-7" },         // 0.95 -> 0.9 (boundary)
      { task: "codegen", model: "claude-haiku-4-5" },   // 0.78 -> 0.8
      { task: "reasoning", model: "llama-3.3-70b" },    // 0.76 -> 0.8
      { task: "qa", model: "llama-3.1-8b" },            // 0.68 -> 0.7
    ];
    for (const c of cases) {
      const measured = predictQuality(c.task, c.model);
      const fromPrior = priorPredictQuality(c.task, c.model);
      expect(Math.abs(measured - fromPrior)).toBeLessThanOrEqual(tick + 1e-9);
    }
  });

  test("predictQualityWithCI reports n=PRIOR_N when serving the prior", () => {
    const ci = predictQualityWithCI("qa", "claude-opus-4-7");
    expect(ci.n).toBe(PRIOR_N);
    expect(ci.mean).toBeGreaterThan(0);
    expect(ci.mean).toBeLessThanOrEqual(1);
    expect(ci.lo95).toBeLessThanOrEqual(ci.mean);
    expect(ci.hi95).toBeGreaterThanOrEqual(ci.mean);
  });

  test("getQualitySourceInfo reports source='prior' when file missing", () => {
    // Trigger a lookup so the predictor loads.
    predictQualityWithCI("qa", "claude-opus-4-7");
    const info = getQualitySourceInfo();
    expect(info.source).toBe("prior");
    expect(info.loadedFrom).toBeUndefined();
  });

  test("unknown model returns DEFAULT_QUALITY=0.5 with a wide CI", () => {
    const ci = predictQualityWithCI("qa", "nonexistent-model");
    expect(ci.mean).toBe(0.5);
    expect(ci.n).toBe(PRIOR_N);
    // The Wilson 95% interval for 5/10 spans roughly [0.24, 0.76]; verify
    // the width is consistent with that.
    expect(ci.lo95).toBeGreaterThan(0.23);
    expect(ci.lo95).toBeLessThan(0.24);
    expect(ci.hi95).toBeGreaterThan(0.76);
    expect(ci.hi95).toBeLessThan(0.77);
  });

  test("ordering relations from the prior survive into the predictor", () => {
    // The seeded prior was hand-tuned so opus > haiku on reasoning. That
    // invariant must hold under the quantization too — otherwise
    // we'd have silently weakened the prior.
    expect(predictQuality("reasoning", "claude-opus-4-7")).toBeGreaterThan(
      predictQuality("reasoning", "claude-haiku-4-5")
    );
  });
});

// ---------------------------------------------------------------------------
// Measured path: file present
// ---------------------------------------------------------------------------

const writeQualityFile = (cells: Record<string, Record<string, { successes: number; trials: number }>>): string => {
  tmpRoot = mkdtempSync(join(tmpdir(), "routerlab-quality-"));
  const path = join(tmpRoot, "quality_table.json");
  const file = {
    schema_version: 1,
    generated_at: "2026-05-10T00:00:00Z",
    cells,
  };
  writeFileSync(path, JSON.stringify(file, null, 2), "utf8");
  return path;
};

describe("predictQuality* — measured table on disk", () => {
  test("returns the measured mean when a cell is present", () => {
    const path = writeQualityFile({
      "claude-opus-4-7": {
        qa: { successes: 95, trials: 100 },
      },
    });
    process.env[QUALITY_TABLE_PATH_ENV_VAR] = path;
    __resetQualityCacheForTest();

    const ci = predictQualityWithCI("qa", "claude-opus-4-7");
    expect(ci.mean).toBe(0.95);
    expect(ci.n).toBe(100);
    expect(ci.lo95).toBeGreaterThan(0);
    expect(ci.lo95).toBeLessThan(ci.mean);
    expect(ci.hi95).toBeGreaterThan(ci.mean);
    expect(ci.hi95).toBeLessThanOrEqual(1);
  });

  test("Wilson CI on the measured cell narrows as n grows", () => {
    // Two files, same proportion (90%), different sample sizes. The
    // larger n must produce a strictly narrower CI.
    const pathSmall = writeQualityFile({
      "claude-opus-4-7": { qa: { successes: 9, trials: 10 } },
    });
    process.env[QUALITY_TABLE_PATH_ENV_VAR] = pathSmall;
    __resetQualityCacheForTest();
    const small = predictQualityWithCI("qa", "claude-opus-4-7");

    rmSync(tmpRoot!, { recursive: true, force: true });
    const pathBig = writeQualityFile({
      "claude-opus-4-7": { qa: { successes: 900, trials: 1000 } },
    });
    process.env[QUALITY_TABLE_PATH_ENV_VAR] = pathBig;
    __resetQualityCacheForTest();
    const big = predictQualityWithCI("qa", "claude-opus-4-7");

    expect(big.mean).toBeCloseTo(0.9, 5);
    expect(small.mean).toBeCloseTo(0.9, 5);
    expect(big.hi95 - big.lo95).toBeLessThan(small.hi95 - small.lo95);
  });

  test("cells outside the measured table fall back to the prior", () => {
    const path = writeQualityFile({
      "claude-opus-4-7": { qa: { successes: 95, trials: 100 } },
      // Note: codegen for opus is intentionally not present.
    });
    process.env[QUALITY_TABLE_PATH_ENV_VAR] = path;
    __resetQualityCacheForTest();

    const present = predictQualityWithCI("qa", "claude-opus-4-7");
    expect(present.n).toBe(100); // measured

    const missingTask = predictQualityWithCI("codegen", "claude-opus-4-7");
    // The model is in the measured table but the task isn't, so the
    // predictor returns the uniform fallback for that cell. `n` therefore
    // equals `PRIOR_N`, not 100.
    expect(missingTask.n).toBe(PRIOR_N);
    expect(missingTask.mean).toBe(0.5);
  });

  test("getQualitySourceInfo reports source='measured' with metadata", () => {
    const path = writeQualityFile({
      "claude-opus-4-7": { qa: { successes: 95, trials: 100 } },
    });
    process.env[QUALITY_TABLE_PATH_ENV_VAR] = path;
    __resetQualityCacheForTest();

    predictQualityWithCI("qa", "claude-opus-4-7");
    const info = getQualitySourceInfo();
    expect(info.source).toBe("measured");
    expect(info.loadedFrom).toBe(path);
    expect(info.generatedAt).toBe("2026-05-10T00:00:00Z");
  });

  test("malformed file with no usable cells silently falls back to prior", () => {
    tmpRoot = mkdtempSync(join(tmpdir(), "routerlab-quality-"));
    const path = join(tmpRoot, "quality_table.json");
    // Schema-shaped object but every cell is invalid (negative trials).
    writeFileSync(
      path,
      JSON.stringify({
        schema_version: 1,
        cells: {
          "claude-opus-4-7": { qa: { successes: -1, trials: -5 } },
        },
      }),
      "utf8"
    );
    process.env[QUALITY_TABLE_PATH_ENV_VAR] = path;
    __resetQualityCacheForTest();

    const ci = predictQualityWithCI("qa", "claude-opus-4-7");
    // Fallback prior is in effect, so `n` is the synthetic PRIOR_N.
    expect(ci.n).toBe(PRIOR_N);
    expect(getQualitySourceInfo().source).toBe("prior");
  });

  test("invalid JSON throws a descriptive error", () => {
    tmpRoot = mkdtempSync(join(tmpdir(), "routerlab-quality-"));
    const path = join(tmpRoot, "quality_table.json");
    writeFileSync(path, "this is not json", "utf8");
    process.env[QUALITY_TABLE_PATH_ENV_VAR] = path;
    __resetQualityCacheForTest();

    expect(() => predictQualityWithCI("qa", "claude-opus-4-7")).toThrow(
      /not valid JSON/
    );
  });
});

// ---------------------------------------------------------------------------
// Memoization
// ---------------------------------------------------------------------------

describe("predictQuality* — memoization", () => {
  test("repeat calls without reset return the cached result", () => {
    const path = writeQualityFile({
      "claude-opus-4-7": { qa: { successes: 95, trials: 100 } },
    });
    process.env[QUALITY_TABLE_PATH_ENV_VAR] = path;
    __resetQualityCacheForTest();

    const a = predictQualityWithCI("qa", "claude-opus-4-7");

    // Overwrite the file. The predictor's cache should not notice.
    writeFileSync(
      path,
      JSON.stringify({
        schema_version: 1,
        generated_at: "2026-05-10T00:00:00Z",
        cells: {
          "claude-opus-4-7": { qa: { successes: 10, trials: 100 } },
        },
      }),
      "utf8"
    );

    const b = predictQualityWithCI("qa", "claude-opus-4-7");
    expect(a.mean).toBe(b.mean);
    expect(a.n).toBe(b.n);
    expect(a.lo95).toBe(b.lo95);
    expect(a.hi95).toBe(b.hi95);
  });

  test("reset re-reads disk on the next lookup", () => {
    const path = writeQualityFile({
      "claude-opus-4-7": { qa: { successes: 95, trials: 100 } },
    });
    process.env[QUALITY_TABLE_PATH_ENV_VAR] = path;
    __resetQualityCacheForTest();

    const a = predictQualityWithCI("qa", "claude-opus-4-7");
    expect(a.mean).toBe(0.95);

    writeFileSync(
      path,
      JSON.stringify({
        schema_version: 1,
        generated_at: "2026-05-10T00:00:00Z",
        cells: {
          "claude-opus-4-7": { qa: { successes: 10, trials: 100 } },
        },
      }),
      "utf8"
    );
    __resetQualityCacheForTest();

    const b = predictQualityWithCI("qa", "claude-opus-4-7");
    expect(b.mean).toBe(0.1);
  });

  test("concurrent reads converge to the same memoized result", async () => {
    const path = writeQualityFile({
      "claude-opus-4-7": { qa: { successes: 80, trials: 100 } },
    });
    process.env[QUALITY_TABLE_PATH_ENV_VAR] = path;
    __resetQualityCacheForTest();

    // Kick off 32 concurrent lookups. Each one calls into `getQualityTable`
    // which is synchronous and memoized via a module-scoped variable, so
    // there is no async race condition here — but the test verifies the
    // behaviour matches the documented contract under heavy load.
    const results: QualityWithCI[] = await Promise.all(
      Array.from({ length: 32 }, () =>
        Promise.resolve(predictQualityWithCI("qa", "claude-opus-4-7"))
      )
    );
    const first = results[0]!;
    for (const r of results) {
      expect(r.mean).toBe(first.mean);
      expect(r.n).toBe(first.n);
      expect(r.lo95).toBe(first.lo95);
      expect(r.hi95).toBe(first.hi95);
    }
  });
});

// ---------------------------------------------------------------------------
// Backward-compat: predictQuality plain signature
// ---------------------------------------------------------------------------

describe("predictQuality — backward-compat scalar signature", () => {
  test("returns the mean from predictQualityWithCI", () => {
    const path = writeQualityFile({
      "claude-opus-4-7": { qa: { successes: 73, trials: 100 } },
    });
    process.env[QUALITY_TABLE_PATH_ENV_VAR] = path;
    __resetQualityCacheForTest();

    const scalar = predictQuality("qa", "claude-opus-4-7");
    const ci = predictQualityWithCI("qa", "claude-opus-4-7");
    expect(scalar).toBe(ci.mean);
    expect(scalar).toBe(0.73);
  });

  test("unknown model returns 0.5", () => {
    expect(predictQuality("qa", "totally-fake-model")).toBe(0.5);
  });
});
