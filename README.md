# routerlab

> Cost-quality routing for LLM APIs with reproducible Pareto frontiers per task class.

## What this is

`routerlab` is an open-source library + CLI that routes each LLM task to the cheapest model that meets a quality threshold. Cost is grounded in real token economics (via [tokenometer](https://github.com/faraa2m/tokenometer)) and quality is predicted **before** the call, not measured after. Per-task Pareto frontiers are published openly so anyone can pick a model rationally.

Where existing routers tend to hand-wave cost or hide their methodology, routerlab is cost-first, reproducible, and open end-to-end.

## Status

The `@routerlab/core` and `@routerlab/cli` packages are published and usable
for early production experiments. Engine, gateway, and eval APIs are still
evolving, but documented behavior is tested and release-managed through
Changesets.

## Install

```bash
npm install @routerlab/core
npm install --save-dev @routerlab/cli
```

## Usage

```bash
# Route a single prompt at a quality bar of 0.85 for QA tasks:
npx --yes @routerlab/cli route --task=qa --quality-bar=0.85 --input=prompt.txt
```

Programmatic:

```ts
import { route } from "@routerlab/core";

const decision = await route({ task: "qa", qualityBar: 0.85, prompt });
// => { model, expectedCost, expectedQuality, fallback }
```

## Adoption examples

- [Production routing examples](./examples/README.md) show support chatbot,
  extraction, summarization, and code-review routing patterns.
- [OpenAI-compatible gateway design](./docs/GATEWAY.md) shows how to put
  routerlab in front of existing SDK clients without changing application
  prompts.
- The CLI package docs in [`packages/cli`](./packages/cli/README.md) cover
  `npx --yes @routerlab/cli route`, `frontier`, `models`, and eval commands.

## Reproducing the published frontier

```bash
bun install
bun run eval:all     # regenerates eval/results/frontier.json + plots
```

Cached judge outputs and provider responses keep this affordable (default judge is the cheapest competent model in the candidate pool).

## Candidate pool

- **Anthropic:** Opus 4.7, Sonnet 4.6, Haiku 4.5.
- **Free-tier:** Groq (Llama 3.3 70B, Llama 3.1 8B, Mixtral 8x7B), Together, HuggingFace Inference, OpenRouter.

## Citation

```bibtex
@misc{routerlab-2026,
  author       = {Faraazuddin Mohammed},
  title        = {{routerlab}: Practical Cost-Quality Routing for LLM APIs},
  year         = {2026},
  howpublished = {\url{https://github.com/faraa2m/routerlab}}
}
```

## Project Health

- [Contributing guide](./CONTRIBUTING.md)
- [Security policy](./SECURITY.md)
- [Code of Conduct](./CODE_OF_CONDUCT.md)
- [Changelog](./CHANGELOG.md)

## License

[Apache-2.0](./LICENSE)
