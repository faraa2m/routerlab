// errors.ts — canonical CLI exit codes + a typed error for graceful failures.
//
// Exit-code semantics (mirrored in the CLI's README so users can script
// against them):
//
//   0 — success
//   1 — no candidates pass the filters (engine ran, decided nothing routable)
//   2 — invalid input (bad flag, malformed value, missing required arg)
//   3 — downstream error (calibration file malformed, missing file on disk,
//       provider runner failure, etc.)
//
// The CLI never throws raw `Error` to the user. Every user-visible failure
// is funneled through `CliError`, which carries the exit code and a clean
// message. `main()` catches and renders.

export const EXIT_SUCCESS = 0;
export const EXIT_NO_CANDIDATES = 1;
export const EXIT_INVALID_INPUT = 2;
export const EXIT_DOWNSTREAM = 3;

export type CliExitCode =
  | typeof EXIT_SUCCESS
  | typeof EXIT_NO_CANDIDATES
  | typeof EXIT_INVALID_INPUT
  | typeof EXIT_DOWNSTREAM;

/**
 * The one error type the CLI throws for expected, user-facing failures.
 * Unknown exceptions still bubble up to `main()` and are mapped to
 * `EXIT_DOWNSTREAM` with the underlying message preserved.
 */
export class CliError extends Error {
  readonly code: CliExitCode;

  constructor(code: CliExitCode, message: string) {
    super(message);
    this.name = "CliError";
    this.code = code;
  }
}
