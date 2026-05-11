// router.ts — the routing engine. Given a task and a quality bar, pick the
// cheapest model that meets the bar and the caller's budget/latency caps.
//
// Differentiation from prior art:
//
//   1. Cost is grounded in *empirical* token economics, not offline
//      tokenizer proxies. The atlas dataset (project 1 of this milestone)
//      provides per-(provider, model, format) calibration between offline
//      counters and empirical counts. RouterArena, RouteLLM, and other
//      prior routers either use proxy "cost thresholds" in [0,1]
//      (RouteLLM) or `offline_tokenizer_estimate * published_price`
//      (RouterArena). Our cost line is `atlas_calibrated_token_count *
//      published_price` — strictly more accurate.
//
//      The atlas-grounded math lives in `cost.ts` (`estimateCost()`),
//      imported below. `cost.ts` wraps tokenometer's empirical counters
//      and atlas's per-(provider, model) calibration table; this router
//      uses it for both the context-window check (`inputTokens`) and the
//      budget check (`totalUsd`) so the two filters cannot drift.
//
//   2. Every decision returns a full triple — `chosen`, `fallbacks`, and
//      `skipped` (with reasons). This makes the engine debuggable and
//      reproducible: a downstream consumer can replay a decision and see
//      exactly which constraints filtered each candidate. RouteLLM
//      returns only a routing logit; LiteLLM's complexity router returns
//      only a chosen model. Routerlab returns the full decision trace.
//
//   3. The default candidate pool is shipped *with* the package and is
//      versioned alongside the engine in `candidates.json`. Callers can
//      override it, but the default is reproducible and citable.
//
// No external network calls and no LLM invocations happen here. The
// engine produces decisions; the per-provider runners under `eval/runners`
// turn decisions into completions.

import candidatesData from "./candidates.json" with { type: "json" };
import { estimateCost } from "./cost.ts";
// Phase 3: predictor reads from eval/results/quality_table.json when present
// and falls back to the seeded prior in quality_prior.ts otherwise. The
// `predictQuality` signature is intentionally unchanged so the router
// engine didn't need to be rewritten.
import { predictQuality } from "./quality_predictor.ts";
import type {
  ModelCandidate,
  RouteDecision,
  RouteFallback,
  RoutePick,
  RouteRequest,
  RouteSkipped,
} from "./types.ts";

/**
 * Internal shape of `candidates.json`. Kept loose because the JSON is
 * authored by hand and validated at load time.
 */
interface CandidateFile {
  candidates: ModelCandidate[];
}

const DEFAULT_CANDIDATES: readonly ModelCandidate[] = Object.freeze(
  (candidatesData as CandidateFile).candidates.map((c) => Object.freeze({ ...c }))
);

/**
 * Return the default candidate pool. Useful for callers that want to
 * inspect the pool, filter it, or extend it before routing.
 */
export function getDefaultCandidates(): readonly ModelCandidate[] {
  return DEFAULT_CANDIDATES;
}

interface ScoredCandidate {
  model: ModelCandidate;
  expectedCost: number;
  expectedQuality: number;
}

/**
 * Resolve which candidate pool to use for this request. Defaults to the
 * shipped pool; callers can override with `request.candidates`.
 */
function resolveCandidates(request: RouteRequest): readonly ModelCandidate[] {
  if (request.candidates !== undefined) {
    return request.candidates;
  }
  return DEFAULT_CANDIDATES;
}

/**
 * Validate a request and throw a descriptive error if it's malformed. We
 * fail loud here so misconfigured callers see the bug immediately rather
 * than getting a silently-bad routing decision.
 */
function validateRequest(request: RouteRequest): void {
  if (!Number.isFinite(request.qualityBar)) {
    throw new Error("route(): qualityBar must be a finite number");
  }
  if (request.qualityBar < 0 || request.qualityBar > 1) {
    throw new Error("route(): qualityBar must be in [0, 1]");
  }
  if (typeof request.prompt !== "string") {
    throw new Error("route(): prompt must be a string");
  }
  if (
    request.maxCostUsd !== undefined &&
    (!Number.isFinite(request.maxCostUsd) || request.maxCostUsd < 0)
  ) {
    throw new Error("route(): maxCostUsd must be a non-negative finite number");
  }
  if (
    request.maxLatencyMs !== undefined &&
    (!Number.isFinite(request.maxLatencyMs) || request.maxLatencyMs < 0)
  ) {
    throw new Error("route(): maxLatencyMs must be a non-negative finite number");
  }
}

/**
 * Filter the candidate pool by quality bar, budget, and context window.
 * Skipped candidates are returned alongside survivors with a recorded
 * reason — this is what makes routing decisions auditable.
 */
function filterCandidates(
  request: RouteRequest,
  pool: readonly ModelCandidate[]
): { kept: ScoredCandidate[]; skipped: RouteSkipped[] } {
  const kept: ScoredCandidate[] = [];
  const skipped: RouteSkipped[] = [];

  for (const model of pool) {
    const estimate = estimateCost({
      prompt: request.prompt,
      model: model.model,
      provider: model.provider,
      pricing: model.pricing,
      taskClass: request.task,
    });

    if (estimate.inputTokens > model.contextWindow) {
      skipped.push({
        model,
        reason: `prompt requires ~${estimate.inputTokens} input tokens, exceeds context window of ${model.contextWindow}`,
      });
      continue;
    }

    const expectedQuality = predictQuality(request.task, model.model);
    if (expectedQuality < request.qualityBar) {
      skipped.push({
        model,
        reason: `expected quality ${expectedQuality.toFixed(3)} for task "${request.task}" is below quality bar ${request.qualityBar.toFixed(3)}`,
      });
      continue;
    }

    if (request.maxCostUsd !== undefined && estimate.totalUsd > request.maxCostUsd) {
      skipped.push({
        model,
        reason: `expected cost $${estimate.totalUsd.toFixed(6)} exceeds budget $${request.maxCostUsd.toFixed(6)}`,
      });
      continue;
    }

    kept.push({ model, expectedCost: estimate.totalUsd, expectedQuality });
  }

  return { kept, skipped };
}

const MAX_FALLBACKS = 3;

/**
 * Compose the winning pick + fallback list from a sorted-by-cost array.
 *
 * Sort tie-breaker: when expected costs are equal, prefer the higher
 * expected quality. This makes routing deterministic and biases toward
 * better-quality picks at no extra cost — defensible default for a
 * cost-first router.
 */
function selectPickAndFallbacks(
  scored: ScoredCandidate[],
  request: RouteRequest
): { chosen: RoutePick; fallbacks: RouteFallback[] } {
  const sorted = [...scored].sort((a, b) => {
    if (a.expectedCost !== b.expectedCost) {
      return a.expectedCost - b.expectedCost;
    }
    return b.expectedQuality - a.expectedQuality;
  });

  const winner = sorted[0];
  if (winner === undefined) {
    throw new Error("internal: selectPickAndFallbacks called with empty array");
  }

  const chosen: RoutePick = {
    model: winner.model,
    expectedCost: winner.expectedCost,
    expectedQuality: winner.expectedQuality,
    reasoning: buildReasoning(winner, sorted.length, request),
  };

  const fallbacks: RouteFallback[] = sorted
    .slice(1, 1 + MAX_FALLBACKS)
    .map((c, idx) => ({
      model: c.model,
      reason: `cost-rank ${idx + 2} of ${sorted.length} qualifying candidates: expected cost $${c.expectedCost.toFixed(6)}, expected quality ${c.expectedQuality.toFixed(3)}`,
    }));

  return { chosen, fallbacks };
}

function buildReasoning(
  winner: ScoredCandidate,
  poolSize: number,
  request: RouteRequest
): string {
  return [
    `cheapest model meeting quality bar ${request.qualityBar.toFixed(3)} for task "${request.task}"`,
    `of ${poolSize} qualifying candidates`,
    `expected cost $${winner.expectedCost.toFixed(6)} (atlas-grounded estimator)`,
    `expected quality ${winner.expectedQuality.toFixed(3)} (Phase 3 predictor; measured cells when available, seeded prior otherwise)`,
  ].join("; ");
}

/**
 * The public routing entrypoint.
 *
 * Pipeline:
 *   1. Validate the request.
 *   2. Resolve the candidate pool (caller override > shipped default).
 *   3. Filter out candidates that fail the quality bar, the cost budget,
 *      or the model's context window. Record reasons.
 *   4. Sort survivors by expected cost ascending (quality breaks ties).
 *   5. Pick the cheapest survivor as `chosen`; the next three as
 *      `fallbacks`. Anything below that is in `skipped` only if it failed
 *      a constraint — extra cheap-survivors past the fallback list are
 *      simply not returned.
 *   6. If no candidate survives, throw with the full skipped list so the
 *      caller can see exactly what went wrong.
 *
 * Sync return: this function does no I/O. Returning a promise would be
 * misleading.
 */
export function route(request: RouteRequest): RouteDecision {
  validateRequest(request);

  const pool = resolveCandidates(request);
  if (pool.length === 0) {
    throw new Error(
      "route(): candidate pool is empty; pass `candidates` in the request or use the default pool"
    );
  }

  const { kept, skipped } = filterCandidates(request, pool);

  if (kept.length === 0) {
    const reasons = skipped.map((s) => `  - ${s.model.model}: ${s.reason}`).join("\n");
    throw new Error(
      `route(): no candidates passed filtering for task "${request.task}" at quality bar ${request.qualityBar}.\nSkipped:\n${reasons}`
    );
  }

  const { chosen, fallbacks } = selectPickAndFallbacks(kept, request);

  return { chosen, fallbacks, skipped };
}
