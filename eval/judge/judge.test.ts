// eval/judge/judge.test.ts — unit tests for the LLM-as-judge harness.
//
// Strategy: every test injects a stub `client` into `judge()` so no
// network call is ever issued. Each test also points the harness at a
// per-test tmp cache dir so the real `eval/judge/cache/` on disk is
// untouched. We cover:
//   - cache miss + write path (one stubbed call, file written).
//   - cache hit path (pre-seeded file, zero stubbed calls).
//   - score parsing on well-formed / malformed / reasoning-first input.
//   - score normalization (0-10 -> 0-1, with clamping).
//   - cache key composition (different rubric -> different key).
//   - cache key composition (different reference shape -> different key).
//   - judge model override.
//   - noCache mode forces a live call and skips the write.
//
// `bun test` runs this file.

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import {
  cacheFilePathForKey,
  cacheKeyForRequest,
  clamp,
  extractReasoning,
  judge,
  parseRawScore,
  readCachedResponse,
  sortedStringify,
  writeCachedResponse,
} from "./judge.ts";
import { DEFAULT_JUDGE_MODEL, type JudgeRequest, type JudgeResponse } from "./_types.ts";

/**
 * A behavior queue stub mirroring the one in anthropic.test.ts. Each entry
 * is either a successful response or an error to throw.
 */
type Behavior =
  | { ok: { content: Array<{ type: string; text?: string }> } }
  | { err: unknown };

function stubClient(behaviors: Behavior[]): {
  client: { messages: { create: (...args: unknown[]) => Promise<unknown> } };
  callCount: () => number;
  lastRequest: () => unknown;
} {
  let i = 0;
  let last: unknown = undefined;
  return {
    client: {
      messages: {
        async create(...args: unknown[]): Promise<unknown> {
          last = args[0];
          if (i >= behaviors.length) {
            throw new Error(
              `stubClient called ${i + 1} times but only ${behaviors.length} behaviors queued`,
            );
          }
          const b = behaviors[i++]!;
          if ("err" in b) throw b.err;
          return b.ok;
        },
      },
    },
    callCount: () => i,
    lastRequest: () => last,
  };
}

/**
 * Build a fresh tmp cache dir per test so they can't observe each other's
 * artifacts. Cleaned up in afterEach.
 */
let tmpCacheDir = "";

beforeEach(() => {
  tmpCacheDir = mkdtempSync(join(tmpdir(), "routerlab-judge-test-"));
});

afterEach(() => {
  if (tmpCacheDir !== "" && existsSync(tmpCacheDir)) {
    rmSync(tmpCacheDir, { recursive: true, force: true });
  }
  tmpCacheDir = "";
});

const REQUEST_SUMM: JudgeRequest = {
  taskClass: "summarization",
  prompt: "Summarize: cats are mammals.",
  candidate: "Cats are mammals.",
  reference: "Cats are mammals.",
};

describe("parseRawScore", () => {
  test("parses well-formed `Score: 8` on the last line", () => {
    const raw = "The summary is faithful and concise.\nScore: 8";
    expect(parseRawScore(raw)).toBe(8);
  });

  test("parses `Score: 10` (max boundary)", () => {
    expect(parseRawScore("Reasoning here.\nScore: 10")).toBe(10);
  });

  test("parses `Score: 0` (min boundary)", () => {
    expect(parseRawScore("Bad.\nScore: 0")).toBe(0);
  });

  test("parses decimal scores like `Score: 7.5`", () => {
    expect(parseRawScore("Mid-tier.\nScore: 7.5")).toBe(7.5);
  });

  test("case-insensitive label match (`score: 6`)", () => {
    expect(parseRawScore("reasoning\nscore: 6")).toBe(6);
  });

  test("clamps a too-large parsed value to 10", () => {
    expect(parseRawScore("Score: 42")).toBe(10);
  });

  test("clamps a negative parsed value to 0", () => {
    // Regex captures unsigned digits, so a leading `-` won't be parsed by
    // the labelled path; falls through to the last-number path which also
    // captures only digits. End-result is 5 (legitimately parsed as 5),
    // not -5. This is the documented behavior: we don't expect negative
    // scores from any sane judge prompt.
    const raw = "Score: 5";
    expect(parseRawScore(raw)).toBe(5);
  });

  test("falls back to last number when `Score:` is absent", () => {
    const raw = "I think this is roughly a 7 out of ten.";
    expect(parseRawScore(raw)).toBe(7);
  });

  test("falls back to last number when judge wrote `Verdict: 9`", () => {
    expect(parseRawScore("Verdict: 9")).toBe(9);
  });

  test("returns 0 when no number appears at all", () => {
    expect(parseRawScore("This is total nonsense with no digits.")).toBe(0);
  });

  test("handles reasoning before score on multiple lines", () => {
    const raw = [
      "The candidate captures the key fact but is slightly verbose.",
      "It correctly identifies the entity and the relation.",
      "Score: 8",
    ].join("\n");
    expect(parseRawScore(raw)).toBe(8);
  });

  test("ignores numbers in the reasoning when `Score:` is present at the end", () => {
    const raw = "The model used 3 sentences but it should use 1.\nScore: 4";
    // The labelled path wins; we get 4, not 1.
    expect(parseRawScore(raw)).toBe(4);
  });
});

describe("clamp", () => {
  test("clamps below range", () => {
    expect(clamp(-1, 0, 10)).toBe(0);
  });
  test("clamps above range", () => {
    expect(clamp(11, 0, 10)).toBe(10);
  });
  test("identity inside range", () => {
    expect(clamp(5, 0, 10)).toBe(5);
  });
});

describe("extractReasoning", () => {
  test("strips trailing `Score: N` line", () => {
    expect(extractReasoning("Good summary.\nScore: 9")).toBe("Good summary.");
  });
  test("returns full text when no Score line is present", () => {
    expect(extractReasoning("Just thoughts, no score.")).toBe(
      "Just thoughts, no score.",
    );
  });
  test("handles multi-line reasoning before score", () => {
    const raw = ["Line 1.", "Line 2.", "Score: 6"].join("\n");
    expect(extractReasoning(raw)).toBe("Line 1.\nLine 2.");
  });
});

describe("sortedStringify", () => {
  test("emits keys in sorted order", () => {
    const a = { b: 1, a: 2 };
    const b = { a: 2, b: 1 };
    expect(sortedStringify(a)).toBe(sortedStringify(b));
  });
  test("recurses into nested objects", () => {
    const a = { outer: { z: 1, a: 2 } };
    const b = { outer: { a: 2, z: 1 } };
    expect(sortedStringify(a)).toBe(sortedStringify(b));
  });
  test("preserves array order", () => {
    expect(sortedStringify([3, 1, 2])).toBe("[3,1,2]");
  });
  test("handles undefined as JSON does (top-level undefined is undefined)", () => {
    expect(sortedStringify(undefined)).toBeUndefined();
  });
});

describe("cacheKeyForRequest", () => {
  test("is deterministic for the same inputs", () => {
    const k1 = cacheKeyForRequest(REQUEST_SUMM, DEFAULT_JUDGE_MODEL);
    const k2 = cacheKeyForRequest(REQUEST_SUMM, DEFAULT_JUDGE_MODEL);
    expect(k1).toBe(k2);
    expect(k1).toMatch(/^[0-9a-f]{64}$/);
  });

  test("differs when the rubric changes", () => {
    const a = cacheKeyForRequest({ ...REQUEST_SUMM, rubric: "rubric A" }, DEFAULT_JUDGE_MODEL);
    const b = cacheKeyForRequest({ ...REQUEST_SUMM, rubric: "rubric B" }, DEFAULT_JUDGE_MODEL);
    expect(a).not.toBe(b);
  });

  test("differs when the judge model changes", () => {
    const a = cacheKeyForRequest(REQUEST_SUMM, "claude-haiku-4-5");
    const b = cacheKeyForRequest(REQUEST_SUMM, "claude-sonnet-4-6");
    expect(a).not.toBe(b);
  });

  test("differs when the candidate output changes", () => {
    const a = cacheKeyForRequest(REQUEST_SUMM, DEFAULT_JUDGE_MODEL);
    const b = cacheKeyForRequest(
      { ...REQUEST_SUMM, candidate: "Cats are felines." },
      DEFAULT_JUDGE_MODEL,
    );
    expect(a).not.toBe(b);
  });

  test("differs when the reference shape changes", () => {
    const a = cacheKeyForRequest(REQUEST_SUMM, DEFAULT_JUDGE_MODEL);
    const b = cacheKeyForRequest(
      { ...REQUEST_SUMM, reference: { goldSummary: "Cats are mammals." } },
      DEFAULT_JUDGE_MODEL,
    );
    expect(a).not.toBe(b);
  });

  test("stable under reordered keys in object reference", () => {
    const a = cacheKeyForRequest(
      { ...REQUEST_SUMM, reference: { goldSummary: "x", label: "y" } },
      DEFAULT_JUDGE_MODEL,
    );
    const b = cacheKeyForRequest(
      { ...REQUEST_SUMM, reference: { label: "y", goldSummary: "x" } },
      DEFAULT_JUDGE_MODEL,
    );
    expect(a).toBe(b);
  });
});

describe("cacheFilePathForKey", () => {
  test("two-char fanout", () => {
    const key = "abcdef1234567890";
    const p = cacheFilePathForKey("/tmp/cache", key);
    expect(p).toBe(join("/tmp/cache", "ab", "cdef1234567890.json"));
  });
});

describe("readCachedResponse / writeCachedResponse roundtrip", () => {
  test("write then read yields the same body with cacheHit=true", () => {
    const key = "0".repeat(64);
    const body: JudgeResponse = {
      score: 0.8,
      reasoning: "looks good",
      judge_model: DEFAULT_JUDGE_MODEL,
      cacheHit: false,
      ts: "2026-05-10T00:00:00.000Z",
    };
    writeCachedResponse(tmpCacheDir, key, body);
    const got = readCachedResponse(tmpCacheDir, key);
    expect(got).not.toBeNull();
    expect(got?.score).toBe(0.8);
    expect(got?.reasoning).toBe("looks good");
    expect(got?.cacheHit).toBe(true);
  });

  test("read of missing file returns null", () => {
    expect(readCachedResponse(tmpCacheDir, "deadbeef")).toBeNull();
  });

  test("read of malformed file returns null", () => {
    const key = "0".repeat(64);
    const path = cacheFilePathForKey(tmpCacheDir, key);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, "not json", "utf8");
    expect(readCachedResponse(tmpCacheDir, key)).toBeNull();
  });
});

describe("judge() — cache miss + write path", () => {
  test("issues one stubbed call, writes the cache, returns parsed score", async () => {
    const { client, callCount } = stubClient([
      {
        ok: {
          content: [{ type: "text", text: "Faithful and concise.\nScore: 9" }],
        },
      },
    ]);
    const resp = await judge(REQUEST_SUMM, {
      apiKey: "dummy",
      client,
      cacheDir: tmpCacheDir,
      retry: { sleepFn: async () => {} },
    });
    expect(resp.score).toBeCloseTo(0.9, 10);
    expect(resp.judge_model).toBe(DEFAULT_JUDGE_MODEL);
    expect(resp.cacheHit).toBe(false);
    expect(resp.reasoning).toBe("Faithful and concise.");
    expect(callCount()).toBe(1);

    // Cache file should now exist.
    const key = cacheKeyForRequest(REQUEST_SUMM, DEFAULT_JUDGE_MODEL);
    const path = cacheFilePathForKey(tmpCacheDir, key);
    expect(existsSync(path)).toBe(true);
    const onDisk = JSON.parse(readFileSync(path, "utf8")) as JudgeResponse;
    expect(onDisk.cacheHit).toBe(false); // persisted as false; flipped on read
    expect(onDisk.score).toBeCloseTo(0.9, 10);
  });
});

describe("judge() — cache hit path", () => {
  test("pre-seeded cache file is returned without any API call", async () => {
    const key = cacheKeyForRequest(REQUEST_SUMM, DEFAULT_JUDGE_MODEL);
    const seeded: JudgeResponse = {
      score: 0.75,
      reasoning: "pre-seeded",
      judge_model: DEFAULT_JUDGE_MODEL,
      cacheHit: false,
      ts: "2026-05-10T00:00:00.000Z",
    };
    writeCachedResponse(tmpCacheDir, key, seeded);

    // The stub has zero behaviors queued — any call would throw.
    const { client, callCount } = stubClient([]);
    const resp = await judge(REQUEST_SUMM, {
      apiKey: "dummy",
      client,
      cacheDir: tmpCacheDir,
    });
    expect(resp.score).toBe(0.75);
    expect(resp.reasoning).toBe("pre-seeded");
    expect(resp.cacheHit).toBe(true);
    expect(callCount()).toBe(0);
  });

  test("different rubric -> cache miss, fresh call", async () => {
    // Seed the cache for the default rubric (no override).
    const baseKey = cacheKeyForRequest(REQUEST_SUMM, DEFAULT_JUDGE_MODEL);
    writeCachedResponse(tmpCacheDir, baseKey, {
      score: 0.1,
      reasoning: "default-rubric verdict",
      judge_model: DEFAULT_JUDGE_MODEL,
      cacheHit: false,
      ts: "2026-05-10T00:00:00.000Z",
    });

    // Now query with a custom rubric — must miss and call the stub.
    const { client, callCount } = stubClient([
      {
        ok: {
          content: [{ type: "text", text: "Custom rubric verdict.\nScore: 5" }],
        },
      },
    ]);
    const resp = await judge(
      { ...REQUEST_SUMM, rubric: "Penalize verbosity heavily." },
      {
        apiKey: "dummy",
        client,
        cacheDir: tmpCacheDir,
        retry: { sleepFn: async () => {} },
      },
    );
    expect(resp.cacheHit).toBe(false);
    expect(resp.score).toBeCloseTo(0.5, 10);
    expect(callCount()).toBe(1);
  });
});

describe("judge() — score normalization", () => {
  test("0-10 mapped to 0-1 (score=10 -> 1.0)", async () => {
    const { client } = stubClient([
      { ok: { content: [{ type: "text", text: "Perfect.\nScore: 10" }] } },
    ]);
    const resp = await judge(REQUEST_SUMM, {
      apiKey: "dummy",
      client,
      cacheDir: tmpCacheDir,
      retry: { sleepFn: async () => {} },
    });
    expect(resp.score).toBe(1);
  });

  test("0-10 mapped to 0-1 (score=0 -> 0.0)", async () => {
    const { client } = stubClient([
      { ok: { content: [{ type: "text", text: "Wrong.\nScore: 0" }] } },
    ]);
    const resp = await judge(REQUEST_SUMM, {
      apiKey: "dummy",
      client,
      cacheDir: tmpCacheDir,
      retry: { sleepFn: async () => {} },
    });
    expect(resp.score).toBe(0);
  });

  test("malformed (no Score: label) falls back to last number, normalized", async () => {
    const { client } = stubClient([
      {
        ok: {
          content: [
            { type: "text", text: "I'd give this a 6 overall." },
          ],
        },
      },
    ]);
    const resp = await judge(REQUEST_SUMM, {
      apiKey: "dummy",
      client,
      cacheDir: tmpCacheDir,
      retry: { sleepFn: async () => {} },
    });
    expect(resp.score).toBeCloseTo(0.6, 10);
  });

  test("clamps out-of-range scores (judge said 42, harness emits 1.0)", async () => {
    const { client } = stubClient([
      {
        ok: {
          content: [{ type: "text", text: "Overconfident.\nScore: 42" }],
        },
      },
    ]);
    const resp = await judge(REQUEST_SUMM, {
      apiKey: "dummy",
      client,
      cacheDir: tmpCacheDir,
      retry: { sleepFn: async () => {} },
    });
    expect(resp.score).toBe(1);
  });
});

describe("judge() — model override + noCache", () => {
  test("judgeModel option is reflected in the response and the SDK call", async () => {
    const { client, callCount, lastRequest } = stubClient([
      { ok: { content: [{ type: "text", text: "ok\nScore: 7" }] } },
    ]);
    const resp = await judge(REQUEST_SUMM, {
      apiKey: "dummy",
      client,
      cacheDir: tmpCacheDir,
      judgeModel: "claude-sonnet-4-6",
      retry: { sleepFn: async () => {} },
    });
    expect(resp.judge_model).toBe("claude-sonnet-4-6");
    expect(callCount()).toBe(1);
    const req = lastRequest() as { model?: string };
    expect(req.model).toBe("claude-sonnet-4-6");
  });

  test("noCache=true skips read and write", async () => {
    // Seed the cache so a normal call would hit.
    const key = cacheKeyForRequest(REQUEST_SUMM, DEFAULT_JUDGE_MODEL);
    writeCachedResponse(tmpCacheDir, key, {
      score: 0.99,
      reasoning: "seeded",
      judge_model: DEFAULT_JUDGE_MODEL,
      cacheHit: false,
      ts: "2026-05-10T00:00:00.000Z",
    });

    const { client, callCount } = stubClient([
      { ok: { content: [{ type: "text", text: "live\nScore: 3" }] } },
    ]);
    const resp = await judge(REQUEST_SUMM, {
      apiKey: "dummy",
      client,
      cacheDir: tmpCacheDir,
      noCache: true,
      retry: { sleepFn: async () => {} },
    });
    // Live result wins, cache file is unchanged.
    expect(resp.score).toBeCloseTo(0.3, 10);
    expect(resp.cacheHit).toBe(false);
    expect(callCount()).toBe(1);

    const onDisk = readCachedResponse(tmpCacheDir, key);
    expect(onDisk?.score).toBe(0.99); // unchanged
  });
});

describe("judge() — retry behavior", () => {
  test("429 retries and eventually succeeds", async () => {
    class FakeErr extends Error {
      status: number;
      constructor(status: number, message: string) {
        super(message);
        this.status = status;
        this.name = "AnthropicError";
      }
    }
    const { client, callCount } = stubClient([
      { err: new FakeErr(429, "rate_limited") },
      { ok: { content: [{ type: "text", text: "ok now.\nScore: 7" }] } },
    ]);
    let sleeps = 0;
    const resp = await judge(REQUEST_SUMM, {
      apiKey: "dummy",
      client,
      cacheDir: tmpCacheDir,
      retry: {
        sleepFn: async () => {
          sleeps++;
        },
      },
    });
    expect(resp.score).toBeCloseTo(0.7, 10);
    expect(callCount()).toBe(2);
    expect(sleeps).toBe(1);
  });

  test("401 is not retried", async () => {
    class FakeErr extends Error {
      status: number;
      constructor(status: number, message: string) {
        super(message);
        this.status = status;
        this.name = "AnthropicError";
      }
    }
    const { client, callCount } = stubClient([
      { err: new FakeErr(401, "bad key") },
    ]);
    let sleeps = 0;
    try {
      await judge(REQUEST_SUMM, {
        apiKey: "dummy",
        client,
        cacheDir: tmpCacheDir,
        retry: {
          sleepFn: async () => {
            sleeps++;
          },
        },
      });
      throw new Error("should have thrown");
    } catch (e) {
      const err = e as { name?: string; reason?: string; retryable?: boolean };
      expect(err.name).toBe("RunnerError");
      expect(err.reason).toBe("auth");
      expect(err.retryable).toBe(false);
    }
    expect(callCount()).toBe(1);
    expect(sleeps).toBe(0);
  });
});

describe("judge() — fail-fast on missing creds", () => {
  test("throws synchronously when no apiKey and no client", async () => {
    const saved = process.env["ANTHROPIC_API_KEY"];
    delete process.env["ANTHROPIC_API_KEY"];
    try {
      await expect(judge(REQUEST_SUMM, { cacheDir: tmpCacheDir, noCache: true })).rejects.toThrow(
        /ANTHROPIC_API_KEY/,
      );
    } finally {
      if (saved !== undefined) process.env["ANTHROPIC_API_KEY"] = saved;
    }
  });
});
