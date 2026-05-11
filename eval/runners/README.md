# `eval/runners/`

Per-provider model runners used by the `routerlab` eval harness. Each
runner implements the same `Runner` interface from
[`_types.ts`](./_types.ts) — the build orchestrator treats them as
interchangeable black boxes that accept a `RunRequest` and return a
`RunResponse` (or throw a typed `RunnerError`).

## Currently implemented

| Provider    | Models                                                                                                                                                  | Pricing                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Anthropic   | `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5`                                                                                              | per `candidates.json` (paid)                                                                  |
| Groq        | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`                                                                                 | free tier (`usdCost: 0`); paid table available                                                |
| Together    | `meta-llama/Llama-3.3-70B-Instruct-Turbo`, `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `Qwen/Qwen2.5-7B-Instruct-Turbo` | catalog pricing per [Together pricing](https://www.together.ai/pricing); free credits applied server-side |
| HuggingFace | `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`, `microsoft/Phi-3.5-mini-instruct`, `Qwen/Qwen2.5-0.5B-Instruct`                  | free serverless tier (`usdCost: 0`); paid Inference Endpoints priced per GPU/hr (not modeled) |
| OpenRouter  | `meta-llama/llama-3.3-70b-instruct:free`, `qwen/qwen-2.5-72b-instruct:free`, `mistralai/mistral-small-3.1-24b-instruct:free`                            | free-tier models (`usdCost: 0`); add paid models with their MTok rate                         |

## Env vars

| Provider    | Env var               |
| ----------- | --------------------- |
| Anthropic   | `ANTHROPIC_API_KEY`   |
| Groq        | `GROQ_API_KEY`        |
| Together    | `TOGETHER_API_KEY`    |
| HuggingFace | `HF_TOKEN`            |
| OpenRouter  | `OPENROUTER_API_KEY`  |

The factory (`_factory.ts`) reads these at construction time and throws if
absent. The smoke runner (`_smoke.ts`) treats missing keys as "skip
cleanly" rather than failure.

## Retry policy (uniform across runners)

- Max 3 attempts.
- Base 1s delay, doubled each retry (`base × 2^attempt`).
- Symmetric ±20% jitter so concurrent workers don't sync on the same rate
  limit window.
- Retried on `rate_limit` (429), `server` (≥500), `timeout` (AbortError).
- Not retried on `auth` (401/403), `bad_request` (4xx other), `unknown`.

The retry helper lives in [`_retry.ts`](./_retry.ts) (no library — plain
async-await loop per the brief).

## Smoke / connectivity check

```bash
bun eval/runners/_smoke.ts
```

Prints one line per provider:

```
[anthropic] OK    model=claude-haiku-4-5 latency=812ms tokens=8/4 cost=$0.000028 output="Hi!"
[groq]      OK    model=llama-3.1-8b-instant latency=420ms tokens=12/3 cost=$0.000000 output="hi there"
```

If a key is missing the line reads `[provider] SKIP (no API key in env)`
and the exit code stays 0. The exit code is 1 iff any present-credentialled
provider failed — the build orchestrator uses that to gate the eval pipeline.

API keys are never printed (the smoke output only mentions whether a key
is `set` or `missing`).

## Pricing source

`_pricing.ts` reads `packages/core/src/candidates.json` directly via
`resolveJsonModule`. Keep that file as the single source of truth — the
runners do not hardcode pricing.

For Groq specifically: free-tier responses report `usdCost: 0` regardless
of the table. Flip `paidMode: true` on `createGroqRunner` to compute cost
from `candidates.json` instead.

Together / HuggingFace / OpenRouter manage their pricing tables in their
own module files (`together.ts`, `hf.ts`, `openrouter.ts`) because their
model identifiers are namespaced (`meta-llama/...`, `qwen/...:free`) and
do not live in `candidates.json`. The hosted-model pool moves faster on
those providers than on Anthropic/Groq, so co-locating the table with the
runner keeps refreshes scoped and reviewable.

## HuggingFace cold-start handling

HF serverless inference returns 503 with
`{"error":"Model X is currently loading","estimated_time":N}` for the
first request after a model has been evicted from memory. The HF runner
detects this body shape (also handles the legacy 200-with-loading
variant) and retries the request OUTSIDE the standard `withRetries`
budget, using a dedicated cold-start schedule (default 30s, 60s). Total
worst-case latency on a cold model is ~90s of cold-start waits plus the
standard retry budget.

Tests override the schedule via `coldStartDelaysMs: [1, 2]` and a no-op
`coldStartSleepFn` so the unit tests stay millisecond-fast.

## Tests

```bash
bun test eval/runners
```

Tests are fully mocked (no live API calls). Each runner has coverage for:

- A successful call producing a valid `RunResponse`.
- A 429 → retry → success path.
- A 401 → `RunnerError { reason: "auth", retryable: false }` path.

## Adding a new runner

1. Implement `Runner` from `_types.ts`. Use the shared `withRetries` and
   the shared error classifier patterns from
   [`anthropic.ts`](./anthropic.ts) / [`groq.ts`](./groq.ts).
2. Add the provider id to `ProviderId` in `_factory.ts` and a case to
   `createRunner` + `hasCredentialsFor`.
3. Add a row to the smoke runner's `SMOKE_MODELS` map and a row to this
   README's "Currently implemented" table.
4. Add a test file `<provider>.test.ts` mirroring the existing test
   structure.

## Files

| File             | Purpose                                                                                 |
| ---------------- | --------------------------------------------------------------------------------------- |
| `_types.ts`      | Shared `RunRequest` / `RunResponse` / `Runner` interface                                |
| `_retry.ts`      | Exponential-backoff retry loop with jitter                                              |
| `_pricing.ts`    | Mirror of `candidates.json` pricing + Groq alias map                                    |
| `_factory.ts`    | `createRunner(provider)` entrypoint                                                     |
| `_smoke.ts`      | Connectivity check across all runners                                                   |
| `anthropic.ts`   | Anthropic SDK-backed runner                                                             |
| `groq.ts`        | Groq OpenAI-compatible REST runner                                                      |
| `together.ts`    | Together AI OpenAI-compatible REST runner                                               |
| `hf.ts`          | HuggingFace serverless Inference runner (with cold-start retries)                       |
| `openrouter.ts`  | OpenRouter OpenAI-compatible REST runner (free-tier-friendly default pool)              |
| `*.test.ts`      | Mocked unit tests per runner                                                            |
