// eval/runners/groq.test.ts — Groq runner unit tests.
//
// Tests inject a stub `fetchFn` so the runner never hits the network.
// The stub returns a `Response`-like object (a real `Response` works in
// Bun, which has fetch built-in) configured per-call.

import { describe, expect, test } from "bun:test";
import { createGroqRunner } from "./groq.ts";
import { __GROQ_CHAT_ENDPOINT } from "./groq.ts";

interface FakeFetchCall {
  url: string;
  headers: Record<string, string>;
  body: string;
}

/**
 * Build a fake `fetch` driven by a sequence of behaviors. Each call
 * consumes the next behavior. Records every (url, headers, body) for
 * assertions about how the runner shaped the request.
 *
 * Behaviors:
 *   - `{ status, body }` to return a response with that status and body.
 *   - `{ throw: <err> }` to throw the given error from inside fetch.
 */
type FetchBehavior =
  | { status: number; body: unknown }
  | { throw: Error };

function stubFetch(behaviors: FetchBehavior[]): {
  fetchFn: typeof fetch;
  calls: FakeFetchCall[];
} {
  let i = 0;
  const calls: FakeFetchCall[] = [];
  // We cast to `typeof fetch` at the end because `fetch`'s full type
  // surface (preconnect, etc.) is more than the runner needs and the
  // intersection type varies across Bun / lib.dom. The runner only ever
  // calls the function form.
  const impl = async (
    input: Parameters<typeof fetch>[0],
    init?: Parameters<typeof fetch>[1],
  ): Promise<Response> => {
    const url =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.toString()
          : (input as { url: string }).url;
    const headers: Record<string, string> = {};
    if (init?.headers !== undefined) {
      const h = init.headers as Record<string, string>;
      for (const k of Object.keys(h)) headers[k] = h[k] as string;
    }
    const body = typeof init?.body === "string" ? init.body : "";
    calls.push({ url, headers, body });

    if (i >= behaviors.length) {
      throw new Error(`stubFetch called ${i + 1} times but only ${behaviors.length} behaviors queued`);
    }
    const b = behaviors[i++]!;
    if ("throw" in b) throw b.throw;
    return new Response(JSON.stringify(b.body), {
      status: b.status,
      headers: { "Content-Type": "application/json" },
    });
  };
  const fetchFn = impl as unknown as typeof fetch;
  return { fetchFn, calls };
}

describe("createGroqRunner", () => {
  test("happy path hits the OpenAI-compatible endpoint and returns free-tier usdCost=0", async () => {
    const { fetchFn, calls } = stubFetch([
      {
        status: 200,
        body: {
          choices: [{ message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 12, completion_tokens: 3 },
        },
      },
    ]);
    const runner = createGroqRunner({ apiKey: "dummy", fetchFn });
    const resp = await runner.run({
      model: "llama-3.1-8b-instant",
      prompt: "say hi",
      maxTokens: 32,
      temperature: 0,
    });
    // Free-tier: usdCost is exactly zero per the brief.
    expect(resp.usdCost).toBe(0);
    expect(resp.inputTokens).toBe(12);
    expect(resp.outputTokens).toBe(3);
    expect(resp.output).toBe("hi");
    // The runner reports the short-form id, which matches `candidates.json`.
    expect(resp.model).toBe("llama-3.1-8b");
    // It hit the OpenAI-compatible endpoint.
    expect(calls.length).toBe(1);
    expect(calls[0]!.url).toBe(__GROQ_CHAT_ENDPOINT);
    // Auth header was set but is not asserted by value (NEVER log keys).
    expect(typeof calls[0]!.headers["Authorization"]).toBe("string");
    // The body uses Groq's canonical long-form model id, not the short form.
    const sent = JSON.parse(calls[0]!.body) as { model: string };
    expect(sent.model).toBe("llama-3.1-8b-instant");
  });

  test("listModels returns the canonical Groq IDs", () => {
    const { fetchFn } = stubFetch([]);
    const runner = createGroqRunner({ apiKey: "dummy", fetchFn });
    expect(runner.listModels()).toEqual([
      "llama-3.3-70b-versatile",
      "llama-3.1-8b-instant",
      "mixtral-8x7b-32768",
    ]);
    expect(runner.provider).toBe("groq");
  });

  test("accepts short-form model id and normalizes to long-form for the API", async () => {
    const { fetchFn, calls } = stubFetch([
      {
        status: 200,
        body: {
          choices: [{ message: { role: "assistant", content: "ok" } }],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        },
      },
    ]);
    const runner = createGroqRunner({ apiKey: "dummy", fetchFn });
    const resp = await runner.run({ model: "llama-3.1-8b", prompt: "hi" });
    expect(resp.model).toBe("llama-3.1-8b");
    expect(resp.usdCost).toBe(0);
    const sent = JSON.parse(calls[0]!.body) as { model: string };
    expect(sent.model).toBe("llama-3.1-8b-instant");
  });

  test("429 is retried and eventually succeeds", async () => {
    const { fetchFn } = stubFetch([
      { status: 429, body: { error: "rate_limit" } },
      { status: 429, body: { error: "still_rate_limit" } },
      {
        status: 200,
        body: {
          choices: [{ message: { role: "assistant", content: "ok after retry" } }],
          usage: { prompt_tokens: 4, completion_tokens: 5 },
        },
      },
    ]);
    let sleepCalls = 0;
    const sleepFn = async (_ms: number): Promise<void> => {
      sleepCalls++;
    };
    const runner = createGroqRunner({
      apiKey: "dummy",
      fetchFn,
      retry: { sleepFn },
    });
    const resp = await runner.run({ model: "llama-3.1-8b-instant", prompt: "hi" });
    expect(resp.output).toBe("ok after retry");
    expect(resp.usdCost).toBe(0);
    expect(sleepCalls).toBe(2);
  });

  test("401 throws a RunnerError with reason 'auth' and is not retryable", async () => {
    const { fetchFn, calls } = stubFetch([
      { status: 401, body: { error: "bad_key" } },
    ]);
    let sleepCalls = 0;
    const sleepFn = async (_ms: number): Promise<void> => {
      sleepCalls++;
    };
    const runner = createGroqRunner({
      apiKey: "dummy",
      fetchFn,
      retry: { sleepFn },
    });
    try {
      await runner.run({ model: "llama-3.1-8b-instant", prompt: "hi" });
      throw new Error("should have thrown");
    } catch (e) {
      const err = e as { name?: string; reason?: string; retryable?: boolean; provider?: string };
      expect(err.name).toBe("RunnerError");
      expect(err.reason).toBe("auth");
      expect(err.retryable).toBe(false);
      expect(err.provider).toBe("groq");
    }
    expect(sleepCalls).toBe(0);
    expect(calls.length).toBe(1);
  });

  test("paid-tier mode computes usdCost from candidates.json", async () => {
    const { fetchFn } = stubFetch([
      {
        status: 200,
        body: {
          choices: [{ message: { role: "assistant", content: "ok" } }],
          usage: { prompt_tokens: 1_000_000, completion_tokens: 1_000_000 },
        },
      },
    ]);
    const runner = createGroqRunner({ apiKey: "dummy", fetchFn, paidMode: true });
    const resp = await runner.run({ model: "llama-3.1-8b-instant", prompt: "hi" });
    // llama-3.1-8b: $0.05 input + $0.08 output per Mtok at 1M/1M each = 0.13
    expect(resp.usdCost).toBeCloseTo(0.13, 6);
  });

  test("factory throws synchronously when no apiKey is supplied", () => {
    const saved = process.env["GROQ_API_KEY"];
    delete process.env["GROQ_API_KEY"];
    try {
      expect(() => createGroqRunner()).toThrow(/GROQ_API_KEY/);
    } finally {
      if (saved !== undefined) process.env["GROQ_API_KEY"] = saved;
    }
  });
});
