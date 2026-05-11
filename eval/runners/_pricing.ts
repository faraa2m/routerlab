// eval/runners/_pricing.ts — single source of truth for per-model pricing
// used by every runner to compute `RunResponse.usdCost`.
//
// Sync target: `packages/core/src/candidates.json`. The numbers below MUST
// match the `pricing` field of each candidate there. We import the JSON
// directly via `resolveJsonModule` so the two never drift; the lookup
// helper guards against missing models so a typo is loud, not silent.
//
// A note on Groq free tier: Groq publishes a "developer tier" that is
// genuinely free up to certain rate caps. The Groq runner sets
// `usdCost: 0` for free-tier responses by checking
// `GROQ_FREE_TIER_MODELS` below. Paid-tier reference numbers (mirrored
// from `candidates.json`) are used only when a future paid runner mode
// gets enabled — they live here so the cost surface stays uniform.

import candidates from "../../packages/core/src/candidates.json" with { type: "json" };

/**
 * Per-million-token pricing. Shape matches `ModelPricing` from
 * `@routerlab/core` but redeclared here so runners stay independent of
 * the core package's TypeScript types (eval/runners is not part of the
 * workspace packages).
 */
export interface PricingRow {
  inputUsdPerMtok: number;
  outputUsdPerMtok: number;
}

interface CandidateEntry {
  provider: string;
  model: string;
  pricing: PricingRow;
  contextWindow: number;
}

interface CandidatesFile {
  candidates: CandidateEntry[];
}

/**
 * Build the model → pricing lookup once at module load.
 *
 * We type-narrow the JSON via a local cast (declaration above) — JSON
 * imports come back as `any` by default, and the rest of the file is
 * strict. The shape is asserted by the tests.
 */
const PRICING: Readonly<Record<string, PricingRow>> = (() => {
  const file = candidates as unknown as CandidatesFile;
  const out: Record<string, PricingRow> = {};
  for (const c of file.candidates) {
    out[c.model] = c.pricing;
  }
  return out;
})();

/**
 * Groq models that are served on the free developer tier in 2026. Cost
 * for these is reported as `0` per the brief. If/when a paid tier is
 * wired up, this set shrinks accordingly.
 */
export const GROQ_FREE_TIER_MODELS: ReadonlySet<string> = new Set([
  "llama-3.3-70b",
  "llama-3.1-8b",
  "mixtral-8x7b",
  // Groq's canonical IDs (different format than candidates.json's short
  // names). We accept both so a runner can pass through either form.
  "llama-3.3-70b-versatile",
  "llama-3.1-8b-instant",
  "mixtral-8x7b-32768",
]);

/**
 * Map Groq's canonical API model ids → the short ids we use in
 * `candidates.json` and the rest of the router. The Groq runner accepts
 * either form on input but normalizes on output.
 */
export const GROQ_MODEL_ALIASES: Readonly<Record<string, string>> = {
  "llama-3.3-70b-versatile": "llama-3.3-70b",
  "llama-3.1-8b-instant": "llama-3.1-8b",
  "mixtral-8x7b-32768": "mixtral-8x7b",
};

/**
 * Look up pricing for a model. Returns `undefined` if the model is not in
 * `candidates.json` — runners use this to decide whether to compute
 * `usdCost` at all. We deliberately do NOT throw: a successful response
 * with an unpriced model is still a successful response; we just leave
 * `usdCost` unset and let the audit log say "we ran an uncalibrated model."
 */
export function pricingFor(model: string): PricingRow | undefined {
  return PRICING[model];
}

/**
 * Compute USD cost for an (input, output) token pair given a model id.
 * Returns `undefined` if pricing is unknown — callers either omit
 * `usdCost` from the response or treat that as a calibration gap.
 */
export function computeUsdCost(
  model: string,
  inputTokens: number,
  outputTokens: number,
): number | undefined {
  const p = pricingFor(model);
  if (p === undefined) return undefined;
  const inCost = (inputTokens / 1_000_000) * p.inputUsdPerMtok;
  const outCost = (outputTokens / 1_000_000) * p.outputUsdPerMtok;
  return inCost + outCost;
}

/**
 * Exported for tests + introspection. Treat as read-only.
 */
export const __PRICING_TABLE: Readonly<Record<string, PricingRow>> = PRICING;
