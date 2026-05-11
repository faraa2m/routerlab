#!/usr/bin/env bun
// eval/runners/_smoke.ts — connectivity check across runners.
//
// Run this with `bun eval/runners/_smoke.ts` to verify each provider is
// reachable with the current env-var credentials. The orchestrator will
// call this as a pre-flight before kicking off a long eval run.
//
// Behavior:
//   - For each known provider, if the env var is set: send a tiny
//     ("say hi") prompt to the cheapest model from that provider, print
//     `provider:model OK latency=…ms cost=$… output="…"`.
//   - If the env var is missing, skip the provider with a clear note.
//   - Never print the API key (just `set` / `missing`).
//
// Exit codes:
//   - 0 if every present-credentialled provider succeeds.
//   - 1 if any present-credentialled provider fails.

import { hasCredentialsFor, createRunner, type ProviderId } from "./_factory.ts";
import type { Runner, RunResponse } from "./_types.ts";

/**
 * Map provider → the cheapest model to use for smoke. Choosing the
 * cheapest minimizes the credit cost of the connectivity check.
 *
 * Notes per provider:
 *   - anthropic / groq / together: per-token paid (or free-tier-capped)
 *     models; pick the smallest in the catalog.
 *   - hf: free-tier serverless; we deliberately pick the smallest model
 *     so cold-start latency is minimized.
 *   - openrouter: stick to a `:free` model so the smoke run never costs
 *     real money.
 */
const SMOKE_MODELS: Record<ProviderId, string> = {
  anthropic: "claude-haiku-4-5",
  groq: "llama-3.1-8b-instant",
  together: "meta-llama/Llama-3.1-8B-Instruct",
  hf: "Qwen/Qwen2.5-0.5B-Instruct",
  openrouter: "meta-llama/llama-3.3-70b-instruct:free",
};

interface SmokeResult {
  provider: ProviderId;
  skipped: boolean;
  ok: boolean;
  /** Set on success: latency / output / cost summary. */
  detail?: string;
  /** Set on failure: error message and reason. */
  error?: string;
}

async function smokeOne(provider: ProviderId): Promise<SmokeResult> {
  if (!hasCredentialsFor(provider)) {
    return { provider, skipped: true, ok: true };
  }
  let runner: Runner;
  try {
    runner = createRunner(provider);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { provider, skipped: false, ok: false, error: `factory-error: ${msg}` };
  }
  try {
    const resp: RunResponse = await runner.run({
      model: SMOKE_MODELS[provider],
      prompt: "say hi",
      maxTokens: 32,
      temperature: 0,
    });
    const outPreview = resp.output.slice(0, 60).replace(/\s+/g, " ");
    const costStr = resp.usdCost === undefined ? "?" : resp.usdCost.toFixed(6);
    return {
      provider,
      skipped: false,
      ok: true,
      detail:
        `model=${resp.model} latency=${Math.round(resp.latencyMs)}ms ` +
        `tokens=${resp.inputTokens ?? "?"}/${resp.outputTokens ?? "?"} ` +
        `cost=$${costStr} output="${outPreview}"`,
    };
  } catch (e) {
    // Never leak the API key in error output. The runner's `RunnerError`
    // messages don't include credentials, but be defensive about
    // stringifying generic errors.
    const msg = e instanceof Error ? e.message : String(e);
    return { provider, skipped: false, ok: false, error: msg };
  }
}

const PROVIDERS: ProviderId[] = [
  "anthropic",
  "groq",
  "together",
  "hf",
  "openrouter",
];

async function main(): Promise<void> {
  console.log("routerlab runner smoke (anthropic + groq + together + hf + openrouter)");
  console.log("======================================================================");
  let anyFail = false;
  for (const p of PROVIDERS) {
    const res = await smokeOne(p);
    if (res.skipped) {
      console.log(`[${p}] SKIP (no API key in env)`);
    } else if (res.ok) {
      console.log(`[${p}] OK    ${res.detail ?? ""}`);
    } else {
      anyFail = true;
      console.log(`[${p}] FAIL  ${res.error ?? "unknown"}`);
    }
  }
  process.exit(anyFail ? 1 : 0);
}

// Run when invoked directly (Bun sets import.meta.main, falls back to argv check).
const isMain = (() => {
  try {
    const meta = import.meta as ImportMeta & { main?: boolean };
    if (typeof meta.main === "boolean") return meta.main;
  } catch {
    /* fall through */
  }
  return process.argv[1]?.endsWith("_smoke.ts") ?? false;
})();

if (isMain) {
  await main();
}
