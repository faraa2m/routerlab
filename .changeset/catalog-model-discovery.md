---
"@routerlab/core": minor
"@routerlab/cli": minor
---

Add catalog-only model discovery without changing the default routed candidate pool.

`@routerlab/core` now exports `getCatalogModels()` for visible unevaluated models, and the CLI adds `route models --catalog` so users can inspect discovery metadata while routing continues to use only evaluated candidates by default.
