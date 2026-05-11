// eval/runners/together.test.ts — Together runner unit tests.
//
// All tests inject a stub `fetchFn` into `createTogetherRunner` so we
// never hit the network. The standard `withRetries` sleep is replaced
// with a no-op `sleepFn` so retries run instantly.
//
// Coverage:
//   - 200 success → returns RunResponse with computed usdCost.
//   - 429 → retried, eventually succeeds, exact attempt count asserted.
//   - 401 → typed RunnerError(auth, retryable=false).
//   - Unsupported model → bad_request, no retry.
//   - Missing API key → factory throws synchronously.

import { describe, expect, test } from "bun:test";
import {
  createTogetherRunner,
  __TOGETHER_PRICING,
  __TOGETHER_ENDPOINT,
} from "./together.ts";

const SUPPORTED_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo";

function makeResponse(status: number, body: unknown): Response {
  return new Response(typeof body === "string" ? body : JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/**
 * Function-only fetch shape. Bun's runtime `typeof fetch` has extra
 * statics (e.g. `preconnect`) that our stubs don't need to implement.
 * We type stubs through this alias and cast at the boundary.
 */
type FetchImpl = (
  input: Parameters<typeof fetch>[0],
  init?: Parameters<typeof fetch>[1],
) => Promise<Response>;
const asFetch = (impl: FetchImpl): typeof fetch =>
  impl as unknown as typeof fetch;

describe("createTogetherRunner", () => {
  test("happy path returns a well-formed RunResponse", async () => {
    let calls = 0;
    const fetchFn = asFetch(async (url) => {
      calls += 1;
      expect(String(url)).toBe(__TOGETHER_ENDPOINT);
      return makeResponse(200, {
        choices: [
          { message: { content: "Hello world." }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 20 },
      });
    });
    const runner = createTogetherRunner({ apiKey: "tg_test", fetchFn });
    const resp = await runner.run({
      model: SUPPORTED_MODEL,
      prompt: "Say hi.",
    });
    expect(resp.model).toBe(SUPPORTED_MODEL);
    expect(resp.output).toBe("Hello world.");
    expect(resp.inputTokens).toBe(10);
    expect(resp.outputTokens).toBe(20);
    const pricing = __TOGETHER_PRICING[SUPPORTED_MODEL]!;
    const expectedCost =
      (10 / 1_000_000) * pricing.inputUsdPerMtok +
      (20 / 1_000_000) * pricing.outputUsdPerMtok;
    expect(resp.usdCost).toBeCloseTo(expectedCost, 12);
    expect(typeof resp.latencyMs).toBe("number");
    expect(resp.latencyMs).toBeGreaterThanOrEqual(0);
    expect(typeof resp.ts).toBe("string");
    expect(calls).toBe(1);
  });

  test("listModels reflects the priced model set", () => {
    const runner = createTogetherRunner({
      apiKey: "tg_test",
      fetchFn: asFetch(async () => makeResponse(200, {})),
    });
    expect(runner.listModels()).toEqual(Object.keys(__TOGETHER_PRICING));
    expect(runner.provider).toBe("together");
  });

  test("rejects unsupported model with a bad_request RunnerError", async () => {
    const runner = createTogetherRunner({
      apiKey: "tg_test",
      fetchFn: asFetch(async () => makeResponse(200, {})),
    });
    try {
      await runner.run({ model: "does-not-exist", prompt: "hi" });
      throw new Error("should have thrown");
    } catch (e) {
      const err = e as {
        name?: string;
        reason?: string;
        retryable?: boolean;
        provider?: string;
      };
      expect(err.name).toBe("RunnerError");
      expect(err.reason).toBe("bad_request");
      expect(err.retryable).toBe(false);
      expect(err.provider).toBe("together");
    }
  });

  test("429 is retried with exponential backoff and eventually succeeds", async () => {
    let calls = 0;
    const fetchFn = asFetch(async () => {
      calls += 1;
      if (calls < 3) {
        return makeResponse(429, { error: { message: "rate limit" } });
      }
      return makeResponse(200, {
        choices: [{ message: { content: "ok after retry" } }],
        usage: { prompt_tokens: 4, completion_tokens: 4 },
      });
    });
    let sleepCalls = 0;
    const sleepFn = async (_ms: number): Promise<void> => {
      sleepCalls += 1;
    };
    const runner = createTogetherRunner({
      apiKey: "tg_test",
      fetchFn,
      retry: { sleepFn },
    });
    const resp = await runner.run({ model: SUPPORTED_MODEL, prompt: "hi" });
    expect(resp.output).toBe("ok after retry");
    expect(calls).toBe(3);
    // Two retries → two sleeps.
    expect(sleepCalls).toBe(2);
  });

  test("401 throws a RunnerError with reason 'auth' and is not retryable", async () => {
    let calls = 0;
    const fetchFn = asFetch(async () => {
      calls += 1;
      return makeResponse(401, { error: { message: "invalid key" } });
    });
    let sleepCalls = 0;
    const sleepFn = async (_ms: number): Promise<void> => {
      sleepCalls += 1;
    };
    const runner = createTogetherRunner({
      apiKey: "tg_wrong",
      fetchFn,
      retry: { sleepFn },
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
      expect(err.provider).toBe("together");
    }
    expect(sleepCalls).toBe(0);
    expect(calls).toBe(1);
  });

  test("factory throws synchronously when no apiKey is supplied", () => {
    const saved = process.env["TOGETHER_API_KEY"];
    delete process.env["TOGETHER_API_KEY"];
    try {
      expect(() => createTogetherRunner()).toThrow(/TOGETHER_API_KEY/);
    } finally {
      if (saved !== undefined) process.env["TOGETHER_API_KEY"] = saved;
    }
  });
});
