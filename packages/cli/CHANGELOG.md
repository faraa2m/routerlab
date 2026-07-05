# @routerlab/cli

## 1.1.0

### Minor Changes

- [#19](https://github.com/faraa2m/routerlab/pull/19) [`dfcfe8f`](https://github.com/faraa2m/routerlab/commit/dfcfe8f6857bf1665d58d2bc274e0d93fff4e98d) Thanks [@faraa2m](https://github.com/faraa2m)! - Add catalog-only model discovery without changing the default routed candidate pool.

  `@routerlab/core` now exports `getCatalogModels()` for visible unevaluated models, and the CLI adds `route models --catalog` so users can inspect discovery metadata while routing continues to use only evaluated candidates by default.

### Patch Changes

- Updated dependencies [[`dfcfe8f`](https://github.com/faraa2m/routerlab/commit/dfcfe8f6857bf1665d58d2bc274e0d93fff4e98d)]:
  - @routerlab/core@1.2.0

## 1.0.2

### Patch Changes

- [`b5d5c04`](https://github.com/faraa2m/routerlab/commit/b5d5c0416e402533951198a08a80192f930992fd) Thanks [@faraa2m](https://github.com/faraa2m)! - Run the published CLI under Node.js and report package versions from package metadata.

- Updated dependencies [[`b5d5c04`](https://github.com/faraa2m/routerlab/commit/b5d5c0416e402533951198a08a80192f930992fd)]:
  - @routerlab/core@1.0.2

## 1.0.1

### Patch Changes

- [#13](https://github.com/faraa2m/routerlab/pull/13) [`b477b8d`](https://github.com/faraa2m/routerlab/commit/b477b8dff642d9144055a3eeb21b3028e0f4d646) Thanks [@faraa2m](https://github.com/faraa2m)! - Polish the repository's open-source project documentation and contributor guidance.

- Updated dependencies [[`b477b8d`](https://github.com/faraa2m/routerlab/commit/b477b8dff642d9144055a3eeb21b3028e0f4d646)]:
  - @routerlab/core@1.0.1

## 1.0.0

### Major Changes

- [#11](https://github.com/faraa2m/routerlab/pull/11) [`7ccbc4d`](https://github.com/faraa2m/routerlab/commit/7ccbc4db88625187ae58652dbbc27a4d35e9362e) Thanks [@faraa2m](https://github.com/faraa2m)! - Require Node.js 26 for npm publishing workflows and local repository development.

### Patch Changes

- Updated dependencies [[`7ccbc4d`](https://github.com/faraa2m/routerlab/commit/7ccbc4db88625187ae58652dbbc27a4d35e9362e)]:
  - @routerlab/core@1.0.0

## 0.0.2

### Patch Changes

- [#6](https://github.com/faraa2m/routerlab/pull/6) [`a217352`](https://github.com/faraa2m/routerlab/commit/a2173527c8c275920e83f192d98cfb7c7ddf50e2) Thanks [@faraa2m](https://github.com/faraa2m)! - Verify changesets automation pipeline. No behavioural change — README copy clarification only.

- Updated dependencies [[`a217352`](https://github.com/faraa2m/routerlab/commit/a2173527c8c275920e83f192d98cfb7c7ddf50e2)]:
  - @routerlab/core@0.0.2
