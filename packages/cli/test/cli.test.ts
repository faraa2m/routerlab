// cli.test.ts — Bun unit tests for `@routerlab/cli`.
//
// Tests import `main()` from `../src/main.ts` directly with an injected
// `CliContext` carrying in-memory stdout/stderr/stdin. We never spawn a
// subprocess: tests assert against captured strings + the returned exit
// code so they finish in milliseconds.
//
// Coverage targets the brief's required cases:
//   - `route` with a fixture prompt prints a decision.
//   - `--json` emits valid JSON.
//   - Invalid `--quality-bar` (>1) exits with code 2.
//   - `models` lists candidates.
//   - `frontier` reads the fixture and renders rows.
//
// Plus a handful of supporting cases (no candidates → exit 1, version,
// unknown subcommand → exit 2, JSON parse errors → exit 3, etc.) that
// raise confidence in the dispatcher without bloating the suite.

import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { Readable, Writable } from "node:stream";
import { resolve as resolvePath } from "node:path";

import { main } from "../src/main.ts";
import type { CliContext } from "../src/io.ts";
import {
  EXIT_DOWNSTREAM,
  EXIT_INVALID_INPUT,
  EXIT_NO_CANDIDATES,
  EXIT_SUCCESS,
} from "../src/errors.ts";

// Make the predictor/cost caches hermetic across the suite: point them at
// non-existent paths so the engine uses its seeded fallback tables instead
// of any local artifact files written by an earlier eval run. We restore
// the original env values after the suite so other suites stay clean.
const HERMETIC_NO_FILE = "/tmp/__routerlab_cli_test_does_not_exist__.json";
const SAVED_ENV = {
  quality: process.env["ROUTERLAB_QUALITY_TABLE_PATH"],
  atlas: process.env["ROUTERLAB_ATLAS_RESULTS_PATH"],
};
beforeAll(() => {
  process.env["ROUTERLAB_QUALITY_TABLE_PATH"] = HERMETIC_NO_FILE;
  process.env["ROUTERLAB_ATLAS_RESULTS_PATH"] = HERMETIC_NO_FILE;
});
afterAll(() => {
  if (SAVED_ENV.quality === undefined) {
    delete process.env["ROUTERLAB_QUALITY_TABLE_PATH"];
  } else {
    process.env["ROUTERLAB_QUALITY_TABLE_PATH"] = SAVED_ENV.quality;
  }
  if (SAVED_ENV.atlas === undefined) {
    delete process.env["ROUTERLAB_ATLAS_RESULTS_PATH"];
  } else {
    process.env["ROUTERLAB_ATLAS_RESULTS_PATH"] = SAVED_ENV.atlas;
  }
});

const FIXTURE_FRONTIER = resolvePath(
  import.meta.dir,
  "fixtures",
  "frontier.json"
);

/**
 * Capture a stream by accumulating writes into a string buffer. Returned
 * tuple is `[writable, getCaptured]` so a test can call the getter at the
 * end of the run to inspect what was emitted.
 */
function captureStream(): { stream: Writable; output: () => string } {
  const chunks: Buffer[] = [];
  const stream = new Writable({
    write(chunk: unknown, _encoding, callback): void {
      if (typeof chunk === "string") {
        chunks.push(Buffer.from(chunk, "utf8"));
      } else if (chunk instanceof Uint8Array) {
        chunks.push(Buffer.from(chunk));
      } else {
        chunks.push(Buffer.from(String(chunk), "utf8"));
      }
      callback();
    },
  });
  return {
    stream,
    output: () => Buffer.concat(chunks).toString("utf8"),
  };
}

/**
 * Build a `CliContext` for a single test run. Pass `stdinText` to simulate
 * a piped prompt; pass an empty string (the default) to simulate no stdin.
 */
function makeContext(
  argv: readonly string[],
  stdinText: string = "",
  envOverrides: NodeJS.ProcessEnv = {}
): {
  ctx: CliContext;
  stdout: () => string;
  stderr: () => string;
} {
  const stdout = captureStream();
  const stderr = captureStream();
  // Readable.from accepts strings; the iterator yields the string as one
  // chunk which `readStdinToString` will consume normally.
  const stdin = stdinText.length > 0 ? Readable.from([stdinText]) : Readable.from([]);

  const ctx: CliContext = {
    argv,
    stdout: stdout.stream,
    stderr: stderr.stream,
    stdin,
    // We deliberately don't inherit process.env here so tests are hermetic;
    // pass overrides explicitly when needed.
    env: { ...envOverrides },
    cwd: process.cwd(),
  };

  return {
    ctx,
    stdout: stdout.output,
    stderr: stderr.output,
  };
}

// ---------------------------------------------------------------------------
// `route` subcommand
// ---------------------------------------------------------------------------

describe("route subcommand", () => {
  test("prints a decision for a fixture prompt", async () => {
    const { ctx, stdout, stderr } = makeContext(
      ["route", "--task=qa", "--quality-bar=0.85"],
      "What is the capital of France?\n"
    );
    const code = await main(ctx);
    expect(code).toBe(EXIT_SUCCESS);
    const out = stdout();
    expect(out).toContain("Decision:");
    // The winner under the seeded prior at qualityBar=0.85 for QA is the
    // cheapest of {opus, sonnet} which is sonnet — but we don't pin the
    // exact model in case the prior table evolves. We do assert the
    // expected-cost and expected-quality labels appear.
    expect(out).toContain("expected cost:");
    expect(out).toContain("expected quality:");
    expect(out).toContain("reasoning:");
    expect(stderr()).toBe("");
  });

  test("--json emits valid parseable JSON", async () => {
    const { ctx, stdout } = makeContext(
      ["route", "--task=qa", "--quality-bar=0.85", "--json"],
      "What is the capital of France?\n"
    );
    const code = await main(ctx);
    expect(code).toBe(EXIT_SUCCESS);
    const out = stdout();
    // Should be parseable as JSON.
    const parsed = JSON.parse(out) as Record<string, unknown>;
    expect(parsed["chosen"]).toBeDefined();
    expect(parsed["fallbacks"]).toBeDefined();
    expect(parsed["skipped"]).toBeDefined();
    const chosen = parsed["chosen"] as Record<string, unknown>;
    expect(chosen["model"]).toBeDefined();
    expect(typeof chosen["expectedCost"]).toBe("number");
    expect(typeof chosen["expectedQuality"]).toBe("number");
  });

  test("invalid --quality-bar (>1) exits with code 2", async () => {
    const { ctx, stderr, stdout } = makeContext(
      ["route", "--task=qa", "--quality-bar=1.5"],
      "x"
    );
    const code = await main(ctx);
    expect(code).toBe(EXIT_INVALID_INPUT);
    expect(stderr()).toContain("quality-bar");
    expect(stdout()).toBe("");
  });

  test("invalid --quality-bar (non-numeric) exits with code 2", async () => {
    const { ctx, stderr } = makeContext(
      ["route", "--task=qa", "--quality-bar=notanumber"],
      "x"
    );
    const code = await main(ctx);
    expect(code).toBe(EXIT_INVALID_INPUT);
    expect(stderr()).toContain("not a finite number");
  });

  test("invalid --task exits with code 2", async () => {
    const { ctx, stderr } = makeContext(
      ["route", "--task=invalidtask", "--quality-bar=0.5"],
      "x"
    );
    const code = await main(ctx);
    expect(code).toBe(EXIT_INVALID_INPUT);
    expect(stderr()).toContain('invalid --task "invalidtask"');
  });

  test("empty prompt exits with code 2", async () => {
    const { ctx, stderr } = makeContext(
      ["route", "--task=qa", "--quality-bar=0.85"],
      "" // no stdin → empty prompt
    );
    const code = await main(ctx);
    expect(code).toBe(EXIT_INVALID_INPUT);
    expect(stderr()).toContain("prompt is empty");
  });

  test("no candidates pass filtering exits with code 1", async () => {
    // qualityBar = 0.99 — nothing in the prior table is >=0.99 for any task.
    const { ctx, stderr } = makeContext(
      ["route", "--task=qa", "--quality-bar=0.99"],
      "some prompt text"
    );
    const code = await main(ctx);
    expect(code).toBe(EXIT_NO_CANDIDATES);
    expect(stderr()).toContain("no candidates passed filtering");
  });

  test("unknown flag exits with code 2", async () => {
    const { ctx, stderr } = makeContext(
      ["route", "--task=qa", "--quality-bar=0.85", "--nope=1"],
      "x"
    );
    const code = await main(ctx);
    expect(code).toBe(EXIT_INVALID_INPUT);
    expect(stderr()).toMatch(/unknown option|Unknown option/i);
  });
});

// ---------------------------------------------------------------------------
// `models` subcommand
// ---------------------------------------------------------------------------

describe("models subcommand", () => {
  test("lists all candidates by default", async () => {
    const { ctx, stdout } = makeContext(["models"]);
    const code = await main(ctx);
    expect(code).toBe(EXIT_SUCCESS);
    const out = stdout();
    // The shipped candidates.json includes at least these three Anthropic
    // models and at least one Groq llama variant.
    expect(out).toContain("claude-opus-4-7");
    expect(out).toContain("claude-haiku-4-5");
    expect(out).toContain("llama-3.3-70b");
    expect(out).toContain("provider");
    expect(out).toContain("model");
  });

  test("--provider filters by provider", async () => {
    const { ctx, stdout } = makeContext(["models", "--provider=anthropic"]);
    const code = await main(ctx);
    expect(code).toBe(EXIT_SUCCESS);
    const out = stdout();
    expect(out).toContain("claude-opus-4-7");
    // Groq models should NOT appear when filtered to Anthropic.
    expect(out).not.toContain("llama-3.3-70b");
  });

  test("--json emits valid JSON with a candidates array", async () => {
    const { ctx, stdout } = makeContext(["models", "--json"]);
    const code = await main(ctx);
    expect(code).toBe(EXIT_SUCCESS);
    const parsed = JSON.parse(stdout()) as { candidates: unknown[] };
    expect(Array.isArray(parsed.candidates)).toBe(true);
    expect(parsed.candidates.length).toBeGreaterThan(0);
  });

  test("--catalog lists unevaluated discovery models separately", async () => {
    const { ctx, stdout } = makeContext(["models", "--catalog", "--json"]);
    const code = await main(ctx);
    expect(code).toBe(EXIT_SUCCESS);
    const parsed = JSON.parse(stdout()) as { catalog: { model: string; evaluated: boolean }[] };
    expect(parsed.catalog.some((entry) => entry.model === "gpt-5.5")).toBe(true);
    expect(parsed.catalog.find((entry) => entry.model === "gpt-5.5")?.evaluated).toBe(false);
  });

  test("invalid --provider exits with code 2", async () => {
    const { ctx, stderr } = makeContext(["models", "--provider=imaginary"]);
    const code = await main(ctx);
    expect(code).toBe(EXIT_INVALID_INPUT);
    expect(stderr()).toContain("invalid --provider");
  });
});

// ---------------------------------------------------------------------------
// `frontier` subcommand
// ---------------------------------------------------------------------------

describe("frontier subcommand", () => {
  test("reads fixture and renders rows for qa", async () => {
    const { ctx, stdout } = makeContext([
      "frontier",
      "--task=qa",
      `--path=${FIXTURE_FRONTIER}`,
    ]);
    const code = await main(ctx);
    expect(code).toBe(EXIT_SUCCESS);
    const out = stdout();
    expect(out).toContain("Pareto frontier — task: qa");
    expect(out).toContain("claude-haiku-4-5");
    expect(out).toContain("llama-3.3-70b");
    expect(out).toContain("claude-sonnet-4-6");
    // Rendered as cost ascending so the cheapest (llama-3.3-70b at
    // $0.000380) should appear above claude-haiku-4-5 ($0.000420).
    const idxLlama = out.indexOf("llama-3.3-70b");
    const idxHaiku = out.indexOf("claude-haiku-4-5");
    expect(idxLlama).toBeLessThan(idxHaiku);
  });

  test("--format=json emits valid parseable JSON", async () => {
    const { ctx, stdout } = makeContext([
      "frontier",
      "--task=codegen",
      "--format=json",
      `--path=${FIXTURE_FRONTIER}`,
    ]);
    const code = await main(ctx);
    expect(code).toBe(EXIT_SUCCESS);
    const parsed = JSON.parse(stdout()) as {
      task: string;
      rows: { model: string }[];
    };
    expect(parsed.task).toBe("codegen");
    expect(parsed.rows.length).toBe(2);
    expect(parsed.rows.some((r) => r.model === "claude-opus-4-7")).toBe(true);
  });

  test("missing task in fixture exits with code 3", async () => {
    const { ctx, stderr } = makeContext([
      "frontier",
      "--task=reasoning",
      `--path=${FIXTURE_FRONTIER}`,
    ]);
    const code = await main(ctx);
    expect(code).toBe(EXIT_DOWNSTREAM);
    expect(stderr()).toContain("no rows for task");
  });

  test("missing fixture file exits with code 3", async () => {
    const { ctx, stderr } = makeContext([
      "frontier",
      "--task=qa",
      "--path=/tmp/this-does-not-exist-routerlab-cli-test.json",
    ]);
    const code = await main(ctx);
    expect(code).toBe(EXIT_DOWNSTREAM);
    expect(stderr()).toContain("cannot read");
  });
});

// ---------------------------------------------------------------------------
// `version` and dispatcher
// ---------------------------------------------------------------------------

describe("version subcommand", () => {
  test("prints two version lines", async () => {
    const { ctx, stdout } = makeContext(["version"]);
    const code = await main(ctx);
    expect(code).toBe(EXIT_SUCCESS);
    const out = stdout();
    expect(out).toContain("@routerlab/cli");
    expect(out).toContain("@routerlab/core");
  });
});

describe("dispatcher", () => {
  test("no args shows help and exits 0", async () => {
    const { ctx, stdout } = makeContext([]);
    const code = await main(ctx);
    expect(code).toBe(EXIT_SUCCESS);
    expect(stdout()).toContain("USAGE");
  });

  test("unknown subcommand exits 2 with helpful message", async () => {
    const { ctx, stderr } = makeContext(["frobnicate"]);
    const code = await main(ctx);
    expect(code).toBe(EXIT_INVALID_INPUT);
    expect(stderr()).toContain('unknown subcommand "frobnicate"');
  });

  test("eval with no sub-subcommand exits 2", async () => {
    const { ctx, stderr } = makeContext(["eval"]);
    const code = await main(ctx);
    expect(code).toBe(EXIT_INVALID_INPUT);
    expect(stderr()).toContain("missing subcommand");
  });

  test("eval frontier with missing runner module exits 3", async () => {
    const { ctx, stderr } = makeContext(
      ["eval", "frontier", "--task=qa"],
      "",
      {
        ROUTERLAB_FRONTIER_RUNNER_MODULE: "/tmp/no-such-runner-routerlab-cli-test.ts",
      }
    );
    const code = await main(ctx);
    expect(code).toBe(EXIT_DOWNSTREAM);
    expect(stderr()).toContain("frontier runner is not available");
  });
});
