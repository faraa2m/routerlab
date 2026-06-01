import { priceUsage, type UsageTokens } from "@tokenometer/core";
import { route } from "./router.ts";
import type { ModelCandidate, RouteDecision, RouteRequest } from "./types.ts";

export type BudgetState = "ok" | "warn" | "exhausted";
export type BudgetStepSource = "actual" | "estimated";

export interface BudgetAwareRouterOptions {
  maxBudgetUsd: number;
  /** Fraction of budget at which routing enters warn/degraded mode. Default 0.8. */
  warnAt?: number;
  /** Optional lower quality bar used once the chain reaches warnAt. */
  degradedQualityBar?: number;
}

export interface BudgetStepRecord {
  source: BudgetStepSource;
  model: ModelCandidate;
  totalUsd: number;
  inputUsd: number;
  cachedInputUsd: number;
  outputUsd: number;
  usage?: UsageTokens;
}

export interface BudgetSnapshot {
  maxBudgetUsd: number;
  spentUsd: number;
  remainingUsd: number;
  percentUsed: number;
  state: BudgetState;
  steps: BudgetStepRecord[];
}

export interface BudgetRouteStepRequest extends Omit<RouteRequest, "maxCostUsd"> {
  /** Optional per-step cap. The effective cap is min(remaining budget, this value). */
  maxStepCostUsd?: number;
}

export interface BudgetRouteStepResult {
  decision: RouteDecision;
  snapshotBefore: BudgetSnapshot;
  effectiveQualityBar: number;
  maxCostUsd: number;
}

export interface RecordActualUsageInput {
  model: ModelCandidate;
  usage: UsageTokens;
}

const DEFAULT_WARN_AT = 0.8;

const assertNonNegativeFinite = (field: string, value: number): void => {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`BudgetAwareRouter: ${field} must be a non-negative finite number`);
  }
};

const cloneStep = (step: BudgetStepRecord): BudgetStepRecord => ({
  ...step,
  model: { ...step.model, pricing: { ...step.model.pricing } },
  ...(step.usage !== undefined ? { usage: { ...step.usage } } : {}),
});

export class BudgetAwareRouter {
  readonly maxBudgetUsd: number;
  readonly warnAt: number;
  readonly degradedQualityBar?: number;
  private spentUsd = 0;
  private steps: BudgetStepRecord[] = [];

  constructor(options: BudgetAwareRouterOptions) {
    assertNonNegativeFinite("maxBudgetUsd", options.maxBudgetUsd);
    if (options.maxBudgetUsd === 0) {
      throw new Error("BudgetAwareRouter: maxBudgetUsd must be greater than 0");
    }
    this.maxBudgetUsd = options.maxBudgetUsd;
    this.warnAt = options.warnAt ?? DEFAULT_WARN_AT;
    if (!Number.isFinite(this.warnAt) || this.warnAt < 0 || this.warnAt > 1) {
      throw new Error("BudgetAwareRouter: warnAt must be in [0, 1]");
    }
    if (options.degradedQualityBar !== undefined) {
      if (
        !Number.isFinite(options.degradedQualityBar) ||
        options.degradedQualityBar < 0 ||
        options.degradedQualityBar > 1
      ) {
        throw new Error("BudgetAwareRouter: degradedQualityBar must be in [0, 1]");
      }
      this.degradedQualityBar = options.degradedQualityBar;
    }
  }

  routeStep(request: BudgetRouteStepRequest): BudgetRouteStepResult {
    const snapshotBefore = this.getSnapshot();
    if (snapshotBefore.remainingUsd <= 0) {
      throw new Error("BudgetAwareRouter: budget is exhausted");
    }

    const effectiveQualityBar =
      snapshotBefore.state === "warn" && this.degradedQualityBar !== undefined
        ? Math.min(request.qualityBar, this.degradedQualityBar)
        : request.qualityBar;
    const maxCostUsd =
      request.maxStepCostUsd !== undefined
        ? Math.min(snapshotBefore.remainingUsd, request.maxStepCostUsd)
        : snapshotBefore.remainingUsd;

    const decision = route({
      ...request,
      maxCostUsd,
      qualityBar: effectiveQualityBar,
    });

    return { decision, effectiveQualityBar, maxCostUsd, snapshotBefore };
  }

  recordActualUsage(input: RecordActualUsageInput): BudgetStepRecord {
    const priced = priceUsage({
      pricing: {
        inputUsdPerMtok: input.model.pricing.inputUsdPerMtok,
        outputUsdPerMtok: input.model.pricing.outputUsdPerMtok,
      },
      usage: input.usage,
    });
    return this.record({
      cachedInputUsd: priced.cachedInputUsd,
      inputUsd: priced.inputUsd,
      model: input.model,
      outputUsd: priced.outputUsd,
      source: "actual",
      totalUsd: priced.totalUsd,
      usage: input.usage,
    });
  }

  recordEstimatedStep(decision: RouteDecision): BudgetStepRecord {
    return this.record({
      cachedInputUsd: 0,
      inputUsd: decision.chosen.expectedCost,
      model: decision.chosen.model,
      outputUsd: 0,
      source: "estimated",
      totalUsd: decision.chosen.expectedCost,
    });
  }

  getSnapshot(): BudgetSnapshot {
    const remainingUsd = this.maxBudgetUsd - this.spentUsd;
    const percentUsed = this.spentUsd / this.maxBudgetUsd;
    let state: BudgetState = "ok";
    if (remainingUsd <= 0) state = "exhausted";
    else if (percentUsed >= this.warnAt) state = "warn";
    return {
      maxBudgetUsd: this.maxBudgetUsd,
      percentUsed,
      remainingUsd,
      spentUsd: this.spentUsd,
      state,
      steps: this.steps.map(cloneStep),
    };
  }

  reset(): void {
    this.spentUsd = 0;
    this.steps = [];
  }

  private record(step: BudgetStepRecord): BudgetStepRecord {
    const cloned = cloneStep(step);
    this.steps.push(cloned);
    this.spentUsd += cloned.totalUsd;
    return cloneStep(cloned);
  }
}
