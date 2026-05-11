// io.ts — IO seam for the CLI.
//
// Every subcommand takes a `CliContext` instead of poking at `process.*`
// globals directly. This is the seam that makes the CLI fully unit-testable
// from `bun test` without spawning a child process: tests instantiate a
// `CliContext` with in-memory streams, pass in argv, and read the captured
// stdout/stderr after the call returns.

import type { Readable, Writable } from "node:stream";

export interface CliContext {
  /** argv slice with the binary name and node entry already removed. */
  argv: readonly string[];
  /** Where normal output goes. Tests inject a `MemoryStream`. */
  stdout: Writable;
  /** Where error and diagnostic output goes. */
  stderr: Writable;
  /**
   * Stdin handle. The `route` subcommand reads from this when `--input`
   * is not supplied. Tests inject a primed `Readable` to simulate piping.
   */
  stdin: Readable;
  /** Environment snapshot. Used to read `ROUTERLAB_*` overrides. */
  env: NodeJS.ProcessEnv;
  /** Working directory. Used to resolve relative `--input` paths. */
  cwd: string;
}

/**
 * Read stdin to completion as a UTF-8 string.
 *
 * Returns an empty string if stdin is a TTY (interactive shell, no pipe)
 * — that way `route --task=qa --quality-bar=0.85` without `--input` and
 * without a piped stdin doesn't hang forever waiting on the user.
 */
export async function readStdinToString(stdin: Readable): Promise<string> {
  // Bun's stdin exposes `isTTY` (matches Node), so a TTY-attached caller
  // doesn't block. Real pipes (`echo "hi" | route ...`) and injected
  // memory streams in tests both have `isTTY` falsy.
  const maybeTty = stdin as Readable & { isTTY?: boolean };
  if (maybeTty.isTTY === true) {
    return "";
  }

  const chunks: Buffer[] = [];
  for await (const chunk of stdin) {
    if (typeof chunk === "string") {
      chunks.push(Buffer.from(chunk, "utf8"));
    } else if (chunk instanceof Uint8Array) {
      chunks.push(Buffer.from(chunk));
    } else {
      // Defensive: shouldn't happen on Node/Bun stdin, but keeps the
      // function total-typed under `noImplicitAny`.
      chunks.push(Buffer.from(String(chunk), "utf8"));
    }
  }
  return Buffer.concat(chunks).toString("utf8");
}

/**
 * Convenience: write a string + newline to a stream. Centralizing this
 * keeps the subcommands free of `\n` boilerplate and lets us swap the
 * underlying writer (e.g. if we ever want to color-tag stderr).
 */
export function writeLine(stream: Writable, line: string): void {
  stream.write(line + "\n");
}
