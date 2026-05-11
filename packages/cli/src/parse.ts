// parse.ts — shared argument validation helpers.
//
// Every subcommand parses flags with `node:util` `parseArgs`. The helpers
// here turn the raw `string | undefined` values into validated, typed
// values, throwing `CliError(EXIT_INVALID_INPUT, ...)` on bad input.
//
// Centralizing the validation keeps the subcommands focused on rendering
// and ensures the user gets the same error message for the same kind of
// mistake regardless of which subcommand they're running.

import type { Provider, TaskClass } from "@routerlab/core";

import { CliError, EXIT_INVALID_INPUT } from "./errors.ts";

const VALID_TASKS: ReadonlySet<TaskClass> = new Set([
  "qa",
  "codegen",
  "summarization",
  "classification",
  "reasoning",
]);

const VALID_PROVIDERS: ReadonlySet<Provider> = new Set([
  "anthropic",
  "openai",
  "google",
  "groq",
  "together",
  "hf",
  "openrouter",
]);

const VALID_FORMATS: ReadonlySet<"table" | "json"> = new Set(["table", "json"]);

/**
 * The value type `node:util`'s `parseArgs` returns per flag. We accept the
 * full union (including the array case from `multiple: true`, which the
 * CLI doesn't currently use) so we can swap a flag to multi-value later
 * without rewriting validators. The validators below all reject arrays
 * because none of the current flags are multi-value.
 */
export type ParsedFlagValue = string | boolean | (string | boolean)[] | undefined;

/**
 * Require a string-typed flag value and return it, else throw.
 */
export function requireString(
  flag: string,
  value: ParsedFlagValue
): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new CliError(
      EXIT_INVALID_INPUT,
      `missing required flag --${flag}`
    );
  }
  return value;
}

/**
 * Parse a `--task=...` value into a `TaskClass`, else throw.
 */
export function parseTask(value: ParsedFlagValue): TaskClass {
  const raw = requireString("task", value);
  if (!VALID_TASKS.has(raw as TaskClass)) {
    throw new CliError(
      EXIT_INVALID_INPUT,
      `invalid --task "${raw}". Expected one of: ${[...VALID_TASKS].join(", ")}`
    );
  }
  return raw as TaskClass;
}

/**
 * Parse a `--provider=...` value into a `Provider`, else throw.
 * Unlike `parseTask` this one is optional because `models` accepts no filter.
 */
export function parseProviderOptional(
  value: ParsedFlagValue
): Provider | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string" || value.length === 0) {
    throw new CliError(EXIT_INVALID_INPUT, `--provider expects a value`);
  }
  if (!VALID_PROVIDERS.has(value as Provider)) {
    throw new CliError(
      EXIT_INVALID_INPUT,
      `invalid --provider "${value}". Expected one of: ${[...VALID_PROVIDERS].join(", ")}`
    );
  }
  return value as Provider;
}

/**
 * Parse a `--quality-bar=...` value into a [0, 1] number, else throw.
 *
 * The engine itself also validates this, but doing it here lets us return
 * exit code 2 (invalid input) consistently rather than depending on the
 * engine's exception bubbling up to the catch-all that returns 3.
 */
export function parseQualityBar(value: ParsedFlagValue): number {
  const raw = requireString("quality-bar", value);
  const n = Number(raw);
  if (!Number.isFinite(n)) {
    throw new CliError(
      EXIT_INVALID_INPUT,
      `invalid --quality-bar "${raw}": not a finite number`
    );
  }
  if (n < 0 || n > 1) {
    throw new CliError(
      EXIT_INVALID_INPUT,
      `invalid --quality-bar "${raw}": must be in [0, 1]`
    );
  }
  return n;
}

/**
 * Parse an optional numeric flag like `--max-cost-usd=0.005`. Returns
 * `undefined` if the flag is absent.
 */
export function parseNonNegativeFloatOptional(
  flag: string,
  value: ParsedFlagValue
): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string" || value.length === 0) {
    throw new CliError(EXIT_INVALID_INPUT, `--${flag} expects a value`);
  }
  const n = Number(value);
  if (!Number.isFinite(n)) {
    throw new CliError(
      EXIT_INVALID_INPUT,
      `invalid --${flag} "${value}": not a finite number`
    );
  }
  if (n < 0) {
    throw new CliError(
      EXIT_INVALID_INPUT,
      `invalid --${flag} "${value}": must be non-negative`
    );
  }
  return n;
}

/**
 * Parse an optional positive integer flag like `--n=20`.
 */
export function parsePositiveIntOptional(
  flag: string,
  value: ParsedFlagValue
): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string" || value.length === 0) {
    throw new CliError(EXIT_INVALID_INPUT, `--${flag} expects a value`);
  }
  const n = Number(value);
  if (!Number.isInteger(n) || n <= 0) {
    throw new CliError(
      EXIT_INVALID_INPUT,
      `invalid --${flag} "${value}": must be a positive integer`
    );
  }
  return n;
}

/**
 * Parse a `--format=table|json` value with a sensible default.
 */
export function parseFormat(
  value: ParsedFlagValue,
  defaultValue: "table" | "json" = "table"
): "table" | "json" {
  if (value === undefined) return defaultValue;
  if (typeof value !== "string" || !VALID_FORMATS.has(value as "table" | "json")) {
    throw new CliError(
      EXIT_INVALID_INPUT,
      `invalid --format "${String(value)}". Expected one of: ${[...VALID_FORMATS].join(", ")}`
    );
  }
  return value as "table" | "json";
}
