# @routerlab/core

## 1.2.0

### Minor Changes

- [#19](https://github.com/faraa2m/routerlab/pull/19) [`dfcfe8f`](https://github.com/faraa2m/routerlab/commit/dfcfe8f6857bf1665d58d2bc274e0d93fff4e98d) Thanks [@faraa2m](https://github.com/faraa2m)! - Add catalog-only model discovery without changing the default routed candidate pool.

  `@routerlab/core` now exports `getCatalogModels()` for visible unevaluated models, and the CLI adds `route models --catalog` so users can inspect discovery metadata while routing continues to use only evaluated candidates by default.

## 1.1.0

### Minor Changes

- [#17](https://github.com/faraa2m/routerlab/pull/17) [`2a03146`](https://github.com/faraa2m/routerlab/commit/2a03146aba6193280c33a52218089a99bc0eed0e) Thanks [@faraa2m](https://github.com/faraa2m)! - Add `BudgetAwareRouter` for multi-step agent loops with remaining-budget preflight routing, actual Tokenometer usage pricing, estimated-step fallback accounting, and optional degraded-quality routing after a warning threshold.

## 1.0.2

### Patch Changes

- [`b5d5c04`](https://github.com/faraa2m/routerlab/commit/b5d5c0416e402533951198a08a80192f930992fd) Thanks [@faraa2m](https://github.com/faraa2m)! - Run the published CLI under Node.js and report package versions from package metadata.

## 1.0.1

### Patch Changes

- [#13](https://github.com/faraa2m/routerlab/pull/13) [`b477b8d`](https://github.com/faraa2m/routerlab/commit/b477b8dff642d9144055a3eeb21b3028e0f4d646) Thanks [@faraa2m](https://github.com/faraa2m)! - Polish the repository's open-source project documentation and contributor guidance.

## 1.0.0

### Major Changes

- [#11](https://github.com/faraa2m/routerlab/pull/11) [`7ccbc4d`](https://github.com/faraa2m/routerlab/commit/7ccbc4db88625187ae58652dbbc27a4d35e9362e) Thanks [@faraa2m](https://github.com/faraa2m)! - Require Node.js 26 for npm publishing workflows and local repository development.

## 0.0.2

### Patch Changes

- [#6](https://github.com/faraa2m/routerlab/pull/6) [`a217352`](https://github.com/faraa2m/routerlab/commit/a2173527c8c275920e83f192d98cfb7c7ddf50e2) Thanks [@faraa2m](https://github.com/faraa2m)! - Verify changesets automation pipeline. No behavioural change — README copy clarification only.
