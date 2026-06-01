# `@routerlab/core`

The routing engine for [routerlab](https://github.com/faraa2m/routerlab) —
cost-quality routing for LLM APIs with open Pareto frontiers per task class.

`@routerlab/core` picks the cheapest LLM model that meets a quality bar
(and any caller-supplied budget / latency caps) for a given task. Cost is
grounded in atlas-calibrated empirical token economics rather than
chars/4 proxies, and quality is predicted from a published per-task prior
(or measured eval data when available).

This is the library; the [`@routerlab/cli`](https://www.npmjs.com/package/@routerlab/cli)
package wraps it as the `route` command.

## Install

```bash
bun add @routerlab/core
# or
npm install @routerlab/core
```

## Quick API

```ts
import { BudgetAwareRouter, route, predictQuality, estimateCost } from "@routerlab/core";

const decision = await route({
  task: "qa",
  prompt: "What's the capital of France?",
  qualityBar: 0.85,
  maxCostUsd: 0.005,      // optional hard budget cap
  maxLatencyMs: 2000,     // optional hard latency cap
});

console.log(decision.chosen?.model.model);
// e.g. "claude-sonnet-4-6"

for (const fb of decision.fallbacks) console.log("fallback:", fb.model.model);
for (const sk of decision.skipped)  console.log("skipped:",  sk.model.model, sk.reason);
```

### Budget-aware agent loops

```ts
const budget = new BudgetAwareRouter({
  maxBudgetUsd: 0.25,
  warnAt: 0.8,
  degradedQualityBar: 0.65,
});

while (!done) {
  const step = budget.routeStep({
    task: "reasoning",
    prompt: agentContext,
    qualityBar: 0.85,
  });

  const response = await callYourModel(step.decision.chosen.model, agentContext);

  budget.recordActualUsage({
    model: step.decision.chosen.model,
    usage: {
      inputTokens: response.usage.prompt_tokens,
      outputTokens: response.usage.completion_tokens,
    },
  });
}
```

`routeStep()` caps the next decision by the remaining chain budget. After the
provider call, `recordActualUsage()` prices the real usage with Tokenometer's
runtime usage-pricing primitive. If a provider does not return usage, call
`recordEstimatedStep(step.decision)` to account for the selected estimate.

### `RouteDecision` shape

```ts
type RouteDecision = {
  chosen: RoutePick | null;     // null when no candidate passes the filters
  fallbacks: RouteFallback[];   // ordered cheapest-next
  skipped: RouteSkipped[];      // every dropped candidate, with a reason
  request: RouteRequest;        // echoed
};
```

See `types.ts` for the full shape. `RoutePick` carries
`expectedCost`, `expectedQuality`, `reasoning`, and the underlying
`ModelCandidate`.

## What's in here

- **`route(req)`** — top-level routing entry point.
- **`BudgetAwareRouter`** — stateful task budget controller for multi-step
  agent loops. It preflights each step with routerlab and records actual or
  estimated spend after each call.
- **`predictQuality` / `predictQualityWithCI`** — quality predictor; serves
  measured eval data when present, falls back to the seeded prior table
  (Wilson 95% CI exposed via the `WithCI` variant).
- **`estimateCost` / `estimateCostBatch`** — atlas-calibrated cost
  estimation. This is the load-bearing differentiation versus prior open
  routers: token counts come from offline counters scaled by per-provider
  empirical correction factors from
  [llm-tokens-atlas](https://github.com/faraa2m/llm-tokens-atlas), not a
  chars/4 proxy.
- **`getDefaultCandidates()`** — current candidate pool.
- **Types** — `ModelCandidate`, `RouteRequest`, `RouteDecision`,
  `TaskClass`, `Provider`, etc.

## Environment overrides

| Variable                          | Effect                                                                |
| --------------------------------- | --------------------------------------------------------------------- |
| `ROUTERLAB_ATLAS_RESULTS_PATH`    | Atlas calibration file path (read by `cost.ts`)                       |
| `ROUTERLAB_QUALITY_TABLE_PATH`    | Measured quality table path (read by `quality_predictor.ts`)          |

When unset, the engine falls back to seeded defaults shipped in the
package.

## Links

- Main repo: [`routerlab`](https://github.com/faraa2m/routerlab)
- CLI: [`@routerlab/cli`](https://www.npmjs.com/package/@routerlab/cli)
- Cost-calibration dataset: [`llm-tokens-atlas`](https://github.com/faraa2m/llm-tokens-atlas)

## License

Apache-2.0. See [`LICENSE`](./LICENSE).
