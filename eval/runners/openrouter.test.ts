// eval/runners/openrouter.test.ts — OpenRouter runner unit tests.
//
// All tests inject a stub `fetchFn` and a no-op `sleepFn`. We exercise:
//   - 200 success → RunResponse(usdCost: 0 for :free models, populated tokens).
//   - 429 → retried via standard policy, then succeeds.
//   - 401 → typed RunnerError(auth, retryable=false).
//   - Attribution headers + bearer token are sent; API key never appears
//     in the request body.
//   - Unsupported model → bad_request, no retry.
//   - Missing OPENROUTER_API_KEY → factory throws synchronously.

import { describe, expect, test } from "bun:test";
import {
  createOpenRouterRunner,
  __OPENROUTER_PRICING,
  __OPENROUTER_ENDPOINT,
  __OPENROUTER_ATTRIBUTION,
} from "./openrouter.ts";

const SUPPORTED_MODEL = "meta-llama/llama-3.3-70b-instruct:free";

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

describe("createOpenRouterRunner", () => {
  test("happy path returns RunResponse with usdCost 0 for free-tier model", async () => {
    let calls = 0;
    const fetchFn = asFetch(async (url) => {
      calls += 1;
      expect(String(url)).toBe(__OPENROUTER_ENDPOINT);
      return makeResponse(200, {
        choices: [{ message: { content: "hello" } }],
        usage: { prompt_tokens: 4, completion_tokens: 6 },
      });
    });
    const runner = createOpenRouterRunner({ apiKey: "sk-or-test", fetchFn });
    const resp = await runner.run({
      model: SUPPORTED_MODEL,
      prompt: "Say hello.",
    });
    expect(resp.output).toBe("hello");
    expect(resp.inputTokens).toBe(4);
    expect(resp.outputTokens).toBe(6);
    expect(resp.usdCost).toBe(0);
    expect(calls).toBe(1);
  });

  test("sends attribution headers and bearer token; never leaks key in body", async () => {
    let capturedInit: RequestInit | undefined;
    const fetchFn = asFetch(async (_url, init) => {
      capturedInit = init;
      return makeResponse(200, {
        choices: [{ message: { content: "ok" } }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
    });
    const runner = createOpenRouterRunner({ apiKey: "sk-or-test", fetchFn });
    await runner.run({ model: SUPPORTED_MODEL, prompt: "hi" });
    const headers = capturedInit?.headers as
      | Record<string, string>
      | undefined;
    expect(headers).toBeDefined();
    expect(headers?.["Authorization"]).toBe("Bearer sk-or-test");
    expect(headers?.["X-Title"]).toBe(__OPENROUTER_ATTRIBUTION.title);
    expect(headers?.["HTTP-Referer"]).toBe(__OPENROUTER_ATTRIBUTION.referer);
    const bodyStr =
      typeof capturedInit?.body === "string" ? capturedInit.body : "";
    expect(bodyStr).not.toContain("sk-or-test");
  });

  test("listModels reflects the supported free-tier pool", () => {
    const runner = createOpenRouterRunner({
      apiKey: "sk-or-test",
      fetchFn: asFetch(async () => makeResponse(200, {})),
    });
    expect(runner.listModels()).toEqual(Object.keys(__OPENROUTER_PRICING));
    expect(runner.provider).toBe("openrouter");
  });

  test("429 is retried via standard policy and eventually succeeds", async () => {
    let calls = 0;
    const fetchFn = asFetch(async () => {
      calls += 1;
      if (calls < 3) {
        return makeResponse(429, { error: { message: "rate limit" } });
      }
      return makeResponse(200, {
        choices: [{ message: { content: "ok after retry" } }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
    });
    let sleepCalls = 0;
    const runner = createOpenRouterRunner({
      apiKey: "sk-or-test",
      fetchFn,
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
    expect(resp.output).toBe("ok after retry");
    expect(calls).toBe(3);
    expect(sleepCalls).toBe(2);
  });

  test("401 throws a RunnerError with reason 'auth' and is not retryable", async () => {
    const fetchFn = asFetch(async () =>
      makeResponse(401, { error: { message: "invalid key" } }),
    );
    let sleepCalls = 0;
    const runner = createOpenRouterRunner({
      apiKey: "sk-or-wrong",
      fetchFn,
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
      expect(err.provider).toBe("openrouter");
    }
    expect(sleepCalls).toBe(0);
  });

  test("rejects unsupported model with bad_request", async () => {
    const runner = createOpenRouterRunner({
      apiKey: "sk-or-test",
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
    const saved = process.env["OPENROUTER_API_KEY"];
    delete process.env["OPENROUTER_API_KEY"];
    try {
      expect(() => createOpenRouterRunner()).toThrow(/OPENROUTER_API_KEY/);
    } finally {
      if (saved !== undefined) process.env["OPENROUTER_API_KEY"] = saved;
    }
  });
});
