// eval/runners/anthropic.test.ts — Anthropic runner unit tests.
//
// All tests inject a stub `client` into `createAnthropicRunner` so the
// SDK never actually hits the network. The stub mimics
// `Anthropic.messages.create`'s contract:
//   - returns an object shaped like the SDK's `Message` response, or
//   - throws an SDK-like error object with a `status` field.

import { describe, expect, test } from "bun:test";
import { createAnthropicRunner } from "./anthropic.ts";

/**
 * Minimal SDK-like error type, mirroring the shape the runner classifies
 * against. We don't import the SDK's own error classes — keeps the test
 * decoupled from internal SDK details that could shift across versions.
 */
class FakeAnthropicError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "AnthropicError";
  }
}

/**
 * Build a stub `client` whose `messages.create` is driven by a sequence
 * of behaviors. Each call consumes the next behavior. Out-of-bounds calls
 * throw — tests should assert exact call counts.
 *
 * A behavior is either:
 *   - `{ ok: <response> }` to resolve with that response, or
 *   - `{ err: <error> }`   to throw that error.
 */
type Behavior =
  | { ok: { content: Array<{ type: string; text?: string }>; usage?: { input_tokens?: number; output_tokens?: number } } }
  | { err: unknown };

function stubClient(behaviors: Behavior[]): {
  client: { messages: { create: (...args: unknown[]) => Promise<unknown> } };
  callCount: () => number;
} {
  let i = 0;
  return {
    client: {
      messages: {
        async create(): Promise<unknown> {
          if (i >= behaviors.length) {
            throw new Error(`stubClient called ${i + 1} times but only ${behaviors.length} behaviors queued`);
          }
          const b = behaviors[i++]!;
          if ("err" in b) throw b.err;
          return b.ok;
        },
      },
    },
    callCount: () => i,
  };
}

describe("createAnthropicRunner", () => {
  test("happy path returns a well-formed RunResponse", async () => {
    const { client, callCount } = stubClient([
      {
        ok: {
          content: [{ type: "text", text: "Hello, world!" }],
          usage: { input_tokens: 10, output_tokens: 5 },
        },
      },
    ]);
    const runner = createAnthropicRunner({ apiKey: "dummy", client: client as never });
    const resp = await runner.run({
      model: "claude-haiku-4-5",
      prompt: "say hi",
      maxTokens: 64,
      temperature: 0,
    });
    expect(resp.model).toBe("claude-haiku-4-5");
    expect(resp.output).toBe("Hello, world!");
    expect(resp.inputTokens).toBe(10);
    expect(resp.outputTokens).toBe(5);
    // Haiku pricing: $1 / Mtok input + $5 / Mtok output. 10/1e6 + 5/1e6*5 = 1e-5 + 2.5e-5 = 3.5e-5.
    expect(resp.usdCost).toBeCloseTo(3.5e-5, 10);
    expect(typeof resp.latencyMs).toBe("number");
    expect(resp.latencyMs).toBeGreaterThanOrEqual(0);
    expect(typeof resp.ts).toBe("string");
    expect(callCount()).toBe(1);
  });

  test("listModels returns the canonical Anthropic model set", () => {
    const runner = createAnthropicRunner({ apiKey: "dummy", client: { messages: { create: async () => ({}) } } as never });
    expect(runner.listModels()).toEqual([
      "claude-opus-4-7",
      "claude-sonnet-4-6",
      "claude-haiku-4-5",
    ]);
    expect(runner.provider).toBe("anthropic");
  });

  test("rejects unsupported model with a bad_request RunnerError", async () => {
    const runner = createAnthropicRunner({ apiKey: "dummy", client: { messages: { create: async () => ({}) } } as never });
    try {
      await runner.run({ model: "claude-mystery-9000", prompt: "hi" });
      throw new Error("should have thrown");
    } catch (e) {
      const err = e as { name?: string; reason?: string; retryable?: boolean; provider?: string };
      expect(err.name).toBe("RunnerError");
      expect(err.reason).toBe("bad_request");
      expect(err.retryable).toBe(false);
      expect(err.provider).toBe("anthropic");
    }
  });

  test("429 is retried with exponential backoff and eventually succeeds", async () => {
    const { client, callCount } = stubClient([
      { err: new FakeAnthropicError(429, "rate_limited") },
      { err: new FakeAnthropicError(429, "rate_limited_again") },
      {
        ok: {
          content: [{ type: "text", text: "ok after retry" }],
          usage: { input_tokens: 4, output_tokens: 4 },
        },
      },
    ]);

    // Inject a no-op sleep so the test runs instantly.
    let sleepCalls = 0;
    const sleepFn = async (_ms: number): Promise<void> => {
      sleepCalls++;
    };

    const runner = createAnthropicRunner({
      apiKey: "dummy",
      client: client as never,
      retry: { sleepFn },
    });
    const resp = await runner.run({ model: "claude-haiku-4-5", prompt: "hi" });
    expect(resp.output).toBe("ok after retry");
    expect(callCount()).toBe(3);
    // Two retries → two sleeps.
    expect(sleepCalls).toBe(2);
  });

  test("401 throws a RunnerError with reason 'auth' and is not retryable", async () => {
    const { client, callCount } = stubClient([
      { err: new FakeAnthropicError(401, "bad key") },
    ]);
    let sleepCalls = 0;
    const sleepFn = async (_ms: number): Promise<void> => {
      sleepCalls++;
    };
    const runner = createAnthropicRunner({
      apiKey: "dummy",
      client: client as never,
      retry: { sleepFn },
    });
    try {
      await runner.run({ model: "claude-haiku-4-5", prompt: "hi" });
      throw new Error("should have thrown");
    } catch (e) {
      const err = e as { name?: string; reason?: string; retryable?: boolean; provider?: string };
      expect(err.name).toBe("RunnerError");
      expect(err.reason).toBe("auth");
      expect(err.retryable).toBe(false);
      expect(err.provider).toBe("anthropic");
    }
    // No retries on auth — sleep never called.
    expect(sleepCalls).toBe(0);
    expect(callCount()).toBe(1);
  });

  test("factory throws synchronously when no apiKey is supplied and no client is injected", () => {
    // Snapshot + clear env to make the test deterministic.
    const saved = process.env["ANTHROPIC_API_KEY"];
    delete process.env["ANTHROPIC_API_KEY"];
    try {
      expect(() => createAnthropicRunner()).toThrow(/ANTHROPIC_API_KEY/);
    } finally {
      if (saved !== undefined) process.env["ANTHROPIC_API_KEY"] = saved;
    }
  });
});
