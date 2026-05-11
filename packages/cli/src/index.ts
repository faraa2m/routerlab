#!/usr/bin/env bun
// @routerlab/cli — `route` command entrypoint.
//
// This file is the binary's argv shim. Real CLI logic lives in `main.ts`
// (and the per-subcommand modules) so it can be imported by tests without
// the side effect of consuming `process.argv`.
//
// We exit with `main()`'s return code so shell pipelines see the right
// status (0 success, 1 no candidates, 2 invalid input, 3 downstream error).
// See `./errors.ts` for the canonical exit-code definitions.

import { main } from "./main.ts";

const code = await main({
  argv: process.argv.slice(2),
  stdout: process.stdout,
  stderr: process.stderr,
  stdin: process.stdin,
  env: process.env,
  cwd: process.cwd(),
});

process.exit(code);
