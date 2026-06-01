// @routerlab/core — public entrypoint.
//
// The routing engine: pick the cheapest LLM model that meets a quality
// bar and the caller's budget/latency caps. Cost is grounded in
// atlas-calibrated empirical token economics (see README) rather than
// offline tokenizer proxies — this is the differentiation versus
// RouteLLM, RouterArena, NotDiamond, and other prior open routers.

import { readFileSync } from "node:fs";

function readPackageVersion(): string {
  const packageJsonUrl = new URL("../package.json", import.meta.url);
  const packageJson = JSON.parse(readFileSync(packageJsonUrl, "utf8")) as { version?: string };
  return packageJson.version ?? "0.0.0";
}

export const version = readPackageVersion();

export { route, getDefaultCandidates } from "./router.ts";
export { BudgetAwareRouter } from "./budget-router.ts";
export type {
  BudgetAwareRouterOptions,
  BudgetRouteStepRequest,
  BudgetRouteStepResult,
  BudgetSnapshot,
  BudgetState,
  BudgetStepRecord,
  BudgetStepSource,
  RecordActualUsageInput,
} from "./budget-router.ts";

// Quality predictor: serves measured eval data when present, falls
// back to the seeded prior table from `quality_prior.ts` otherwise.
// `predictQuality` keeps its original signature for backward compat;
// consumers wanting confidence intervals call `predictQualityWithCI`.
export {
  __resetQualityCacheForTest,
  getQualitySourceInfo,
  PRIOR_N,
  predictQuality,
  predictQualityWithCI,
  wilsonScore95,
} from "./quality_predictor.ts";
export type { QualityWithCI } from "./quality_predictor.ts";

export type {
  ModelCandidate,
  ModelPricing,
  Provider,
  RouteDecision,
  RouteFallback,
  RoutePick,
  RouteRequest,
  RouteSkipped,
  TaskClass,
} from "./types.ts";

// Atlas-grounded cost estimation. See cost.ts for the load-bearing
// differentiation note: cost is computed from tokenometer's offline counters
// scaled by `llm-tokens-atlas` per-provider empirical correction factors,
// not from a chars/4 proxy x published pricing (which is what RouteLLM /
// RouterArena / commercial routers do).
export {
  CostEstimationError,
  __resetCalibrationCacheForTest,
  estimateCost,
  estimateCostBatch,
} from "./cost.ts";
export type {
  Confidence,
  CostEstimate,
  CostEstimationFailure,
  CostInput,
  CostPricing,
  CostProvider,
  TokenSource,
} from "./cost.ts";
