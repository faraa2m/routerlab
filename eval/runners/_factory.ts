// eval/runners/_factory.ts — single entrypoint to construct any runner.
//
// The orchestrator (Phase 3) imports `createRunner(provider)` and gets back
// a `Runner` it can call without knowing about provider-specific config.
// All env-var validation happens at construction time (fail-fast).
//
// Sibling agent (`router-runners-B`) extends the `provider` union below
// when adding together / hf / openrouter.

import { createAnthropicRunner } from "./anthropic.ts";
import { createGroqRunner } from "./groq.ts";
import { createHfRunner } from "./hf.ts";
import { createOpenRouterRunner } from "./openrouter.ts";
import { createTogetherRunner } from "./together.ts";
import type { Runner } from "./_types.ts";

/**
 * The set of provider ids the factory knows about. Owned by both runner
 * agents: A added anthropic + groq; B added together + hf + openrouter.
 */
export type ProviderId =
  | "anthropic"
  | "groq"
  | "together"
  | "hf"
  | "openrouter";

/**
 * Construct a `Runner` for the given provider. Throws if the required
 * env var is missing — the build orchestrator can catch this and skip the
 * provider rather than discovering the missing key mid-eval.
 */
export function createRunner(provider: ProviderId): Runner {
  switch (provider) {
    case "anthropic":
      return createAnthropicRunner();
    case "groq":
      return createGroqRunner();
    case "together":
      return createTogetherRunner();
    case "hf":
      return createHfRunner();
    case "openrouter":
      return createOpenRouterRunner();
    default: {
      // Exhaustiveness check: if a new ProviderId lands without a case
      // above, this assignment is a type error.
      const _exhaustive: never = provider;
      throw new Error(`Unknown provider: ${String(_exhaustive)}`);
    }
  }
}

/**
 * Check whether the env vars needed for `provider` are present. Used by
 * the smoke runner and by the build orchestrator to gate provider availability
 * without throwing.
 */
export function hasCredentialsFor(provider: ProviderId): boolean {
  switch (provider) {
    case "anthropic":
      return Boolean(process.env["ANTHROPIC_API_KEY"]);
    case "groq":
      return Boolean(process.env["GROQ_API_KEY"]);
    case "together":
      return Boolean(process.env["TOGETHER_API_KEY"]);
    case "hf":
      return Boolean(process.env["HF_TOKEN"]);
    case "openrouter":
      return Boolean(process.env["OPENROUTER_API_KEY"]);
    default: {
      const _exhaustive: never = provider;
      return _exhaustive;
    }
  }
}
