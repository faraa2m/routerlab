// eval/runners/hf.test.ts — HuggingFace Inference runner unit tests.
//
// All tests inject a stub `fetchFn` (and a no-op `coldStartSleepFn` /
// `retry.sleepFn`) so the suite finishes in milliseconds without hitting
// the network. Cold-start delays are overridden via `coldStartDelaysMs`
// to tiny values; the production 30s/60s schedule is exercised by
// `_smoke.ts` instead.
//
// Coverage:
//   - 200 success → RunResponse(usdCost: 0, populated tokens).
//   - 503-with-loading-body → cold-start path retries, then succeeds.
//   - 200-with-loading-body (legacy path) → same cold-start path.
//   - 429 → retried via the standard policy, then succeeds.
//   - 401 → typed RunnerError(auth, retryable=false).
//   - Unsupported model → bad_request, no retry.
//   - Missing HF_TOKEN → factory throws synchronously.

import { describe, expect, test } from "bun:test";
import {
  createHfRunner,
  __HF_MODELS,
  __HF_ENDPOINT_BASE,
} from "./hf.ts";

const SUPPORTED_MODEL = "meta-llama/Llama-3.2-1B-Instruct";

function makeResponse(status: number, body: unknown): Response {
  return new Response(typeof body === "string" ? body : JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/**
 * Function-only fetch shape. Bun's runtime `typeof fetch` has extra
 * statics (e.g. `preconnect`) that our stubs don't need to implement.
 */
type FetchImpl = (
  input: Parameters<typeof fetch>[0],
  init?: Parameters<typeof fetch>[1],
) => Promise<Response>;
const asFetch = (impl: FetchImpl): typeof fetch =>
  impl as unknown as typeof fetch;

describe("createHfRunner", () => {
  test("happy path returns RunResponse with usdCost 0 (free tier)", async () => {
    let calls = 0;
    const fetchFn = asFetch(async (url) => {
      calls += 1;
      expect(String(url).startsWith(__HF_ENDPOINT_BASE)).toBe(true);
      return makeResponse(200, {
        choices: [{ message: { content: "Hi there." } }],
        usage: { prompt_tokens: 5, completion_tokens: 7 },
      });
    });
    const runner = createHfRunner({
      apiKey: "hf_test",
      fetchFn,
      coldStartDelaysMs: [1],
      coldStartSleepFn: async () => {},
      retry: { sleepFn: async () => {} },
    });
    const resp = await runner.run({
      model: SUPPORTED_MODEL,
      prompt: "Say hi.",
    });
    expect(resp.output).toBe("Hi there.");
    expect(resp.usdCost).toBe(0);
    expect(resp.inputTokens).toBe(5);
    expect(resp.outputTokens).toBe(7);
    expect(calls).toBe(1);
  });

  test("listModels reflects the supported pool", () => {
    const runner = createHfRunner({
      apiKey: "hf_test",
      fetchFn: asFetch(async () => makeResponse(200, {})),
    });
    expect(runner.listModels()).toEqual(Object.keys(__HF_MODELS));
    expect(runner.provider).toBe("hf");
  });

  test("503-with-loading-body retries via cold-start path then succeeds", async () => {
    let calls = 0;
    const fetchFn = asFetch(async () => {
      calls += 1;
      if (calls === 1) {
        return makeResponse(503, {
          error: "Model meta-llama/Llama-3.2-1B-Instruct is currently loading",
          estimated_time: 0.001,
        });
      }
      return makeResponse(200, {
        choices: [{ message: { content: "warm now" } }],
        usage: { prompt_tokens: 2, completion_tokens: 3 },
      });
    });
    let coldSleeps = 0;
    const runner = createHfRunner({
      apiKey: "hf_test",
      fetchFn,
      coldStartDelaysMs: [1, 2],
      coldStartSleepFn: async () => {
        coldSleeps += 1;
      },
      retry: { sleepFn: async () => {} },
    });
    const resp = await runner.run({
      model: SUPPORTED_MODEL,
      prompt: "ping",
    });
    expect(resp.output).toBe("warm now");
    expect(calls).toBe(2);
    expect(coldSleeps).toBe(1);
  });

  test("200-with-loading-body is treated as cold-start too", async () => {
    let calls = 0;
    const fetchFn = asFetch(async () => {
      calls += 1;
      if (calls === 1) {
        return makeResponse(200, {
          error: "Model is currently loading",
          estimated_time: 0.001,
        });
      }
      return makeResponse(200, {
        choices: [{ message: { content: "warm" } }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
    });
    let coldSleeps = 0;
    const runner = createHfRunner({
      apiKey: "hf_test",
      fetchFn,
      coldStartDelaysMs: [1],
      coldStartSleepFn: async () => {
        coldSleeps += 1;
      },
      retry: { sleepFn: async () => {} },
    });
    const resp = await runner.run({
      model: SUPPORTED_MODEL,
      prompt: "ping",
    });
    expect(resp.output).toBe("warm");
    expect(calls).toBe(2);
    expect(coldSleeps).toBe(1);
  });

  test("429 is retried via standard policy and eventually succeeds", async () => {
    let calls = 0;
    const fetchFn = asFetch(async () => {
      calls += 1;
      if (calls === 1) {
        return makeResponse(429, { error: "rate limit" });
      }
      return makeResponse(200, {
        choices: [{ message: { content: "ok" } }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
    });
    let sleepCalls = 0;
    const runner = createHfRunner({
      apiKey: "hf_test",
      fetchFn,
      coldStartDelaysMs: [1],
      coldStartSleepFn: async () => {},
      retry: {
        sleepFn: async () => {
          sleepCalls += 1;
        },
      },
    });
    const resp = await runner.run({
      model: SUPPORTED_MODEL,
      prompt: "ping",
    });
    expect(resp.output).toBe("ok");
    expect(calls).toBe(2);
    expect(sleepCalls).toBe(1);
  });

  test("401 throws a RunnerError with reason 'auth' and is not retryable", async () => {
    const fetchFn = asFetch(async () =>
      makeResponse(401, { error: "invalid token" }),
    );
    let sleepCalls = 0;
    const runner = createHfRunner({
      apiKey: "hf_bad",
      fetchFn,
      coldStartDelaysMs: [1],
      coldStartSleepFn: async () => {},
      retry: {
        sleepFn: async () => {
          sleepCalls += 1;
        },
      },
    });
    try {
      await runner.run({ model: SUPPORTED_MODEL, prompt: "hi" });
      throw new Error("should have thrown");
    } catch (e) {
      const err = e as {
        name?: string;
        reason?: string;
        retryable?: boolean;
        provider?: string;
      };
      expect(err.name).toBe("RunnerError");
      expect(err.reason).toBe("auth");
      expect(err.retryable).toBe(false);
      expect(err.provider).toBe("hf");
    }
    expect(sleepCalls).toBe(0);
  });

  test("rejects unsupported model with bad_request", async () => {
    const runner = createHfRunner({
      apiKey: "hf_test",
      fetchFn: asFetch(async () => makeResponse(200, {})),
    });
    try {
      await runner.run({ model: "does-not-exist", prompt: "hi" });
      throw new Error("should have thrown");
    } catch (e) {
      const err = e as { reason?: string; retryable?: boolean };
      expect(err.reason).toBe("bad_request");
      expect(err.retryable).toBe(false);
    }
  });

  test("factory throws synchronously when no apiKey is supplied", () => {
    const saved = process.env["HF_TOKEN"];
    delete process.env["HF_TOKEN"];
    try {
      expect(() => createHfRunner()).toThrow(/HF_TOKEN/);
    } finally {
      if (saved !== undefined) process.env["HF_TOKEN"] = saved;
    }
  });
});
