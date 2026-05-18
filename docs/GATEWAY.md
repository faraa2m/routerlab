# OpenAI-Compatible Gateway Design

routerlab can sit in front of existing SDK callers as a lightweight routing
gateway. The gateway accepts an OpenAI-compatible request shape, maps it to a
routerlab task class, chooses the cheapest model that satisfies policy, and
forwards the request to the selected provider.

## Minimal Request Contract

```http
POST /v1/chat/completions
content-type: application/json
```

```json
{
  "model": "routerlab:auto",
  "messages": [
    { "role": "system", "content": "You are a concise support assistant." },
    { "role": "user", "content": "Summarize this ticket." }
  ],
  "routerlab": {
    "task": "summarization",
    "qualityBar": 0.82,
    "maxCostUsd": 0.01,
    "maxLatencyMs": 3000
  }
}
```

If `routerlab` is omitted, the gateway should use conservative defaults:
`task: "qa"`, `qualityBar: 0.85`, and no hard budget cap.

## Routing Flow

1. Concatenate chat messages into the prompt string used for routing.
2. Call `route({ task, prompt, qualityBar, maxCostUsd, maxLatencyMs })`.
3. Reject with `422` if no candidate meets policy.
4. Forward the original request to the chosen provider/model.
5. Return the provider response plus routing metadata in headers:
   - `x-routerlab-model`
   - `x-routerlab-expected-cost`
   - `x-routerlab-expected-quality`

## Migration Pattern

Existing OpenAI SDK callers only need to change `baseURL` and set
`model: "routerlab:auto"`. Application prompts and message shapes stay the same.

```ts
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.ROUTERLAB_API_KEY,
  baseURL: "https://router.internal.example/v1",
});

await client.chat.completions.create({
  model: "routerlab:auto",
  messages,
});
```

This design is intentionally documented before shipping a hosted gateway so the
library and CLI can remain useful in private infrastructure.
