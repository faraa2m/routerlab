// @routerlab/core — public type surface for the routing engine.
//
// These types are intentionally narrow and dependency-free. They describe
// what a caller passes in to `route()` and what comes back. Everything else
// (cost computation, quality prediction, candidate pool composition) is an
// internal implementation detail of the engine.

/**
 * The set of task classes routerlab routes for. Each class has its own
 * quality prior table (see `quality_prior.ts`) and, downstream, its own
 * Pareto frontier in `eval/results/frontier.json`.
 *
 * Keep this list small and stable — frontier reproducibility hinges on
 * task-class identity surviving across runs.
 */
export type TaskClass =
  | "qa"
  | "codegen"
  | "summarization"
  | "classification"
  | "reasoning";

/**
 * The providers we can route across. Mirrors the candidate pool documented
 * in the routerlab README and the per-provider runners under `eval/runners/`.
 */
export type Provider =
  | "anthropic"
  | "openai"
  | "google"
  | "groq"
  | "together"
  | "hf"
  | "openrouter";

/**
 * Per-million-token pricing for a model. Routerlab is cost-first, so this
 * is the load-bearing field: every routing decision sorts on a cost
 * computed from these numbers × atlas-grounded token estimates.
 */
export interface ModelPricing {
  inputUsdPerMtok: number;
  outputUsdPerMtok: number;
}

/**
 * A single routable model. The pool of these is what `route()` searches
 * over. Callers can override the default pool via `RouteRequest.candidates`.
 */
export interface ModelCandidate {
  provider: Provider;
  /** Model identifier, e.g. "claude-haiku-4-5", "llama-3.3-70b". */
  model: string;
  pricing: ModelPricing;
  /** Maximum context window in tokens. */
  contextWindow: number;
}

/**
 * A routing request. `task` and `qualityBar` are required; everything else
 * is optional and fills in sensible defaults.
 *
 * `qualityBar` is a value in [0,1] interpreted as "minimum acceptable
 * expected quality on this task class." Candidates below the bar are
 * filtered out before cost-sorting.
 */
export interface RouteRequest {
  task: TaskClass;
  prompt: string;
  /** 0..1 — minimum acceptable expected quality. */
  qualityBar: number;
  /** Hard budget cap per request, in USD. */
  maxCostUsd?: number;
  /** Hard latency cap per request, in milliseconds. */
  maxLatencyMs?: number;
  /** Override the default candidate pool. */
  candidates?: ModelCandidate[];
}

/**
 * A picked model with the engine's reasoning trace attached.
 *
 * `expectedCost` is in USD; `expectedQuality` is in [0,1].
 */
export interface RoutePick {
  model: ModelCandidate;
  expectedCost: number;
  expectedQuality: number;
  /** Human-readable explanation of why this model won. */
  reasoning: string;
}

/**
 * A non-chosen candidate retained as a fallback (e.g. for retries on
 * provider failure). Sorted by ascending expected cost.
 */
export interface RouteFallback {
  model: ModelCandidate;
  reason: string;
}

/**
 * A candidate filtered out before cost-sorting, with the failing constraint
 * recorded. This is what makes routerlab debuggable — every rejection has
 * a documented cause.
 */
export interface RouteSkipped {
  model: ModelCandidate;
  reason: string;
}

/**
 * The full return value of `route()`. The triple (chosen, fallbacks,
 * skipped) is a complete decision record: callers can reproduce why a
 * particular model was picked from this object alone.
 */
export interface RouteDecision {
  chosen: RoutePick;
  fallbacks: RouteFallback[];
  skipped: RouteSkipped[];
}
