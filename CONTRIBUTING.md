# Contributing

Thanks for improving routerlab. This project is about transparent cost-quality
routing, so contributions should keep model selection explainable and
reproducible.

## Development

```bash
bun install --frozen-lockfile
bun run --filter '*' build
bun test
```

Run `bun run eval:smoke` for changes that affect routing or evaluation code.

## Pull requests

- Keep changes focused on one routing, gateway, eval, CLI, or docs concern.
- Add tests for model selection, frontier, policy, or CLI behavior changes.
- Document user-facing behavior in the relevant README or `docs/`.
- Add a Changeset for package changes.

## Evaluation changes

Changes to frontiers, task classes, or judge methodology should include enough
metadata for another contributor to reproduce the result.
