# routerlab Production Examples

These examples show where cost-quality routing fits in application code. The
same pattern applies whether the final provider call is made directly, through
an internal gateway, or by an OpenAI-compatible proxy.

## Support Chatbot

Use a higher quality bar for escalations and a lower bar for simple FAQ answers.

```ts
import { route } from "@routerlab/core";

const decision = await route({
  task: "qa",
  prompt: ticketContext,
  qualityBar: ticket.priority === "urgent" ? 0.9 : 0.8,
  maxLatencyMs: 2500,
});

if (!decision.chosen) throw new Error("No model meets the support policy");
```

Store `decision.chosen.model.model`, expected cost, and expected quality next to
the support trace so future evals can compare routing policy against outcomes.

## Structured Extraction

Extraction tasks often tolerate cheaper models when the schema is clear.

```ts
const decision = await route({
  task: "classification",
  prompt: invoiceExtractionPrompt,
  qualityBar: 0.82,
  maxCostUsd: 0.002,
});
```

If no model passes the budget and quality filters, fall back to a manual review
queue rather than silently lowering the quality bar.

## Summarization Pipeline

Batch summarization is where small cost differences compound quickly.

```ts
const decision = await route({
  task: "summarization",
  prompt: longDocumentPrompt,
  qualityBar: 0.78,
});
```

Use Tokenometer before routing when the document can be chunked; route each
chunk class separately if short and long summaries have different risk levels.

## Code Review Assistant

Code review prompts usually deserve a stricter quality threshold and a latency
cap that keeps PR feedback interactive.

```ts
const decision = await route({
  task: "codegen",
  prompt: reviewPrompt,
  qualityBar: 0.88,
  maxLatencyMs: 5000,
});
```

Log skipped candidates. If the chosen model changes after a frontier update,
the skip reasons explain whether the change came from cost, quality, or latency.

## Frontier Snapshot

Include frontier output in docs or dashboards so teams can see the tradeoff:

```bash
npx --yes @routerlab/cli frontier --task=summarization --format=json > frontier.summarization.json
npx --yes @routerlab/cli frontier --task=codegen
```

Publish the generated table or chart alongside routing policy changes. The goal
is to make model choice reviewable instead of hidden in application code.
