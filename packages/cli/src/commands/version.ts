// commands/version.ts — print CLI + core versions.
//
// We import the core version from `@routerlab/core` directly so the two
// numbers can drift independently and the CLI honestly reports both.
// CLI version is read from its own package.json at build time via the
// constant below — keep this in lockstep with packages/cli/package.json.

import { version as coreVersion } from "@routerlab/core";

import { type CliContext, writeLine } from "../io.ts";

// Pinned to packages/cli/package.json. If you bump `package.json`, bump
// this too — the repo's CI catches drift between the two via test.
export const CLI_VERSION = "0.0.1";

/**
 * `version` subcommand: prints two lines, `@routerlab/cli` first then
 * `@routerlab/core`. Returns 0 unconditionally.
 */
export function runVersion(ctx: CliContext): number {
  writeLine(ctx.stdout, `@routerlab/cli ${CLI_VERSION}`);
  writeLine(ctx.stdout, `@routerlab/core ${coreVersion}`);
  return 0;
}
