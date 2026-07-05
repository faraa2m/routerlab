import type { ModelCatalogEntry } from "./types.ts";

const CATALOG_MODELS: readonly ModelCatalogEntry[] = Object.freeze([
  {
    provider: "openai",
    model: "gpt-5.5",
    pricing: { inputUsdPerMtok: 5, outputUsdPerMtok: 30 },
    contextWindow: 400000,
    status: "stable",
    modalities: ["text", "image"],
    sourceUrl: "https://developers.openai.com/api/docs/pricing",
    evaluated: false,
    unsupportedReason: "Not in the routed pool until quality data or seeded priors are added.",
  },
  {
    provider: "openai",
    model: "gpt-5.4",
    pricing: { inputUsdPerMtok: 2.5, outputUsdPerMtok: 15 },
    contextWindow: 400000,
    status: "stable",
    modalities: ["text", "image"],
    sourceUrl: "https://developers.openai.com/api/docs/pricing",
    evaluated: false,
    unsupportedReason: "Not in the routed pool until quality data or seeded priors are added.",
  },
  {
    provider: "openai",
    model: "gpt-5.4-mini",
    pricing: { inputUsdPerMtok: 0.75, outputUsdPerMtok: 4.5 },
    contextWindow: 200000,
    status: "stable",
    modalities: ["text", "image"],
    sourceUrl: "https://developers.openai.com/api/docs/pricing",
    evaluated: false,
    unsupportedReason: "Not in the routed pool until quality data or seeded priors are added.",
  },
  {
    provider: "anthropic",
    model: "claude-fable-5",
    pricing: { inputUsdPerMtok: 10, outputUsdPerMtok: 50 },
    contextWindow: 200000,
    status: "stable",
    modalities: ["text", "image"],
    sourceUrl: "https://platform.claude.com/docs/en/about-claude/pricing",
    evaluated: false,
    unsupportedReason: "Not in the routed pool until quality data or seeded priors are added.",
  },
  {
    provider: "anthropic",
    model: "claude-opus-4-8",
    pricing: { inputUsdPerMtok: 5, outputUsdPerMtok: 25 },
    contextWindow: 200000,
    status: "stable",
    modalities: ["text", "image"],
    sourceUrl: "https://platform.claude.com/docs/en/about-claude/pricing",
    evaluated: false,
    unsupportedReason: "Not in the routed pool until quality data or seeded priors are added.",
  },
  {
    provider: "anthropic",
    model: "claude-sonnet-5",
    pricing: { inputUsdPerMtok: 2, outputUsdPerMtok: 10 },
    contextWindow: 200000,
    status: "stable",
    modalities: ["text", "image"],
    sourceUrl: "https://platform.claude.com/docs/en/about-claude/pricing",
    evaluated: false,
    unsupportedReason: "Not in the routed pool until quality data or seeded priors are added.",
  },
]);

export function getCatalogModels(): readonly ModelCatalogEntry[] {
  return CATALOG_MODELS;
}
