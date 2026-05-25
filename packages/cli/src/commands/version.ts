// commands/version.ts — print CLI + core versions.
//
// We read package.json at runtime so Changesets version bumps are reflected
// without duplicating version constants in source.

import { readFileSync } from "node:fs";

import { type CliContext, writeLine } from "../io.ts";

function readPackageVersion(relativePackageJson: string): string {
  const packageJsonUrl = new URL(relativePackageJson, import.meta.url);
  const packageJson = JSON.parse(readFileSync(packageJsonUrl, "utf8")) as { version?: string };
  return packageJson.version ?? "0.0.0";
}

/**
 * `version` subcommand: prints two lines, `@routerlab/cli` first then
 * `@routerlab/core`. Returns 0 unconditionally.
 */
export function runVersion(ctx: CliContext): number {
  writeLine(ctx.stdout, `@routerlab/cli ${readPackageVersion("../../package.json")}`);
  writeLine(ctx.stdout, `@routerlab/core ${readPackageVersion("../../../core/package.json")}`);
  return 0;
}
