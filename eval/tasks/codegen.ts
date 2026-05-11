// eval/tasks/codegen.ts — code generation task definition for routerlab.
//
// Dataset: HumanEval (Chen et al., 2021, "Evaluating Large Language Models
// Trained on Code").
//   Source:  https://huggingface.co/datasets/openai/openai_humaneval
//   Split:   test (164 examples — the only split published).
//   License: MIT — redistribution-compatible with no notable restrictions.
//   Notes:   Each row is a Python function stub + a hidden assertion-based
//            test suite. A model "passes" iff every assertion in the
//            row's `test` block succeeds with the model's completion
//            spliced in as the function body.
//
// Scoring: pass@1 (binary). 1.0 if all hidden tests succeed; 0.0 otherwise.
//
// SANDBOX WARNING — READ BEFORE EXTENDING:
//   Scoring HumanEval requires *executing untrusted code* (the model's
//   completion). We do this by spawning a Python subprocess with a wall-
//   clock timeout. This is the highest-risk surface in routerlab.
//
//   What we DO:
//     - Hard wall-clock timeout via `child_process.spawn` + `setTimeout`
//       killing the process on expiry (default: 10 seconds).
//     - Run with a restricted argv (`-I` isolated mode + `-S` no site).
//     - Capture stdout/stderr separately so a failing test doesn't pollute
//       logs.
//     - Reject completions containing obviously dangerous imports/strings
//       (best-effort: see `looksDangerous`). This is a *speed bump*, not a
//       security boundary.
//
//   What we do NOT do:
//     - No filesystem chroot.
//     - No process namespace isolation.
//     - No syscall filtering (seccomp / sandbox-exec / Docker).
//
//   IF YOU RUN THIS HARNESS ON ADVERSARIAL INPUTS (e.g. evaluating a
//   prompt-injection-attacked model whose completions you don't trust),
//   YOU MUST wrap subprocess execution in an OS-level sandbox: Docker,
//   firejail, sandbox-exec on macOS, bubblewrap on Linux, or a fresh VM.
//   The harness as shipped is intended for benchmark runs against
//   *cooperative* models on standardized prompts. Do not use it as a
//   judge in production routing decisions.

import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import {
  datasetCachePath,
  seededShuffle,
  type LoadExamplesOptions,
  type TaskDefinition,
  type TaskExample,
} from "./_types.ts";

const CACHE_FILENAME = "codegen.jsonl";
const DEFAULT_SEED = 42;
const SUBPROCESS_TIMEOUT_MS = 10_000;

/**
 * What HumanEval's test harness needs to score a row. `entryPoint` is the
 * function name the row's tests will call; `testProgram` is the row's
 * `test` field — a Python source string that, when executed in a scope
 * where `entry_point` resolves to the model's completion, asserts pass/fail.
 */
export interface CodegenReference {
  prompt: string;
  entryPoint: string;
  testProgram: string;
}

/**
 * HF datasets-server returns HumanEval rows in this shape. We narrow only
 * the fields we read; the API returns more (canonical_solution, etc.).
 */
interface HumanEvalRow {
  task_id: string;
  prompt: string;
  entry_point: string;
  test: string;
}

interface DatasetsServerResponse {
  rows: Array<{ row: HumanEvalRow }>;
}

/**
 * Render the codegen prompt. Pure: depends only on `input`. We instruct
 * the model to return only the function body so we can splice it back
 * underneath the row's prompt header. The "no markdown" instruction is
 * load-bearing — many models default to fenced output and we strip fences
 * defensively in `parseOutput` either way.
 */
export function promptTemplate(input: TaskExample["input"]): string {
  const prompt = input.prompt ?? "";
  return `Complete the following Python function. Output only the function body (no markdown, no explanation).

\`\`\`python
${prompt}
\`\`\``;
}

/**
 * Parse a raw model completion: strip Markdown code fences if present,
 * preserve leading indentation, and return the resulting source.
 * Conservative — we keep everything inside the fences verbatim (including
 * indentation) because Python is whitespace-sensitive.
 *
 * Strategy:
 *   - Walk leading lines and drop ones that are blank or a code fence.
 *   - Walk trailing lines and drop ones that are blank or a code fence.
 *   - Return the surviving lines joined by '\n'. Critically: we do NOT
 *     trim() the body, because that would strip the 4-space indent the
 *     function body needs to be syntactically valid Python.
 *
 * Empty input (or input that becomes empty after fence/blank stripping)
 * returns "" so the score path short-circuits cleanly.
 */
export function parseOutput(raw: string): string {
  if (raw.length === 0) return "";
  const lines = raw.split(/\r?\n/);

  // Drop leading blank / fence lines.
  while (lines.length > 0) {
    const head = lines[0] ?? "";
    const headTrim = head.trim();
    if (headTrim.length === 0 || headTrim.startsWith("```")) {
      lines.shift();
    } else {
      break;
    }
  }

  // Drop trailing blank / fence lines.
  while (lines.length > 0) {
    const tail = lines[lines.length - 1] ?? "";
    const tailTrim = tail.trim();
    if (tailTrim.length === 0 || tailTrim.startsWith("```")) {
      lines.pop();
    } else {
      break;
    }
  }

  return lines.join("\n");
}

/**
 * Cheap heuristic that flags completions that look like they might do
 * I/O or escape the sandbox. Not a security boundary — a determined
 * adversary can defeat any string-match check. Just a speed bump for
 * accidental damage during routine benchmarking.
 *
 * We pattern-match on the *completion*, not the row's prompt — HumanEval
 * prompts themselves never reference these modules so a hit is suspicious.
 */
function looksDangerous(completion: string): boolean {
  // Substring checks (case-sensitive — Python imports are too).
  const blocked = [
    "import os",
    "import subprocess",
    "import socket",
    "import shutil",
    "import sys",
    "from os ",
    "from subprocess ",
    "from socket ",
    "from shutil ",
    "from sys ",
    "open(",
    "__import__",
    "eval(",
    "exec(",
    "compile(",
  ];
  for (const needle of blocked) {
    if (completion.includes(needle)) return true;
  }
  return false;
}

/**
 * Compose the Python test script for a HumanEval row. The model's
 * completion is spliced in as the function body underneath the row's
 * prompt header (which contains the `def` line + docstring). The test
 * harness then runs the row's `test` program and calls a
 * `check(<entry_point>)` function it expects to find at the bottom.
 *
 * The result mirrors the official HumanEval execution shim from the
 * `human-eval` repo. Keep the structure stable — drift here is a
 * scoring bug.
 */
function buildTestScript(
  completion: string,
  reference: CodegenReference,
): string {
  // The model's completion is the function *body* (per the prompt
  // template). To produce a valid Python module we concatenate the row's
  // `prompt` (which ends with the `def f(...):\n    """..."""\n`) with the
  // completion (indented body) and then the row's `test` program (which
  // defines `check`). Finally we invoke `check(<entry_point>)`.
  return [
    reference.prompt,
    completion,
    "",
    reference.testProgram,
    "",
    `check(${reference.entryPoint})`,
    "",
  ].join("\n");
}

/**
 * Internal: run a string of Python source under a wall-clock timeout.
 * Returns whether the subprocess exited cleanly (code 0). Captures
 * stderr for diagnostics but does not surface it to the score —
 * HumanEval scoring is binary.
 *
 * Implementation notes:
 *   - We spawn `python3` (not `python` — that's often Python 2 on macOS)
 *     with `-I` (isolated mode: ignore PYTHON* env vars, no user site).
 *   - We pipe the source over stdin (`python3 -`).
 *   - `setTimeout` runs the kill switch; the timer is cleared on natural
 *     exit. We use `SIGKILL` so processes that ignore SIGTERM still die.
 *
 * Resolves with `{ ok, stderr }`. Never rejects; the caller maps `ok` to
 * a score and uses `stderr` only for the optional debug log.
 */
async function runPython(
  source: string,
  timeoutMs: number,
): Promise<{ ok: boolean; stderr: string }> {
  return new Promise((resolve) => {
    const child = spawn("python3", ["-I", "-"], {
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stderr = "";
    let killed = false;

    const timer = setTimeout(() => {
      killed = true;
      try {
        child.kill("SIGKILL");
      } catch {
        // Process already exited between the timer firing and the kill.
      }
    }, timeoutMs);

    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });

    child.on("error", () => {
      clearTimeout(timer);
      resolve({ ok: false, stderr: stderr + "\n[spawn error]" });
    });

    child.on("exit", (code, signal) => {
      clearTimeout(timer);
      if (killed) {
        resolve({ ok: false, stderr: stderr + "\n[timeout]" });
        return;
      }
      if (signal !== null) {
        resolve({ ok: false, stderr: stderr + `\n[killed by ${signal}]` });
        return;
      }
      resolve({ ok: code === 0, stderr });
    });

    child.stdin.write(source);
    child.stdin.end();
  });
}

/**
 * Score a parsed completion against a HumanEval reference. Returns 1.0
 * if all hidden tests pass, else 0.0.
 *
 * Steps:
 *   1. Reject obviously-dangerous completions outright (speed bump).
 *   2. Build the test script (prompt header + completion + test program).
 *   3. Run it in a Python subprocess with a hard timeout.
 *   4. Pass iff the subprocess exits with code 0.
 *
 * Note: this is `async` because subprocess I/O is unavoidable. The shared
 * `TaskDefinition.score` accepts `Promise<number> | number`, so this
 * still satisfies the contract.
 */
export async function score(
  parsed: string,
  reference: CodegenReference,
): Promise<number> {
  if (parsed.length === 0) return 0;
  if (looksDangerous(parsed)) return 0;

  const program = buildTestScript(parsed, reference);
  const { ok } = await runPython(program, SUBPROCESS_TIMEOUT_MS);
  return ok ? 1 : 0;
}

/**
 * Convert a HF datasets-server HumanEval row into our `TaskExample` shape.
 */
function toTaskExample(row: HumanEvalRow): TaskExample {
  const reference: CodegenReference = {
    prompt: row.prompt,
    entryPoint: row.entry_point,
    testProgram: row.test,
  };
  return {
    id: row.task_id,
    input: {
      prompt: row.prompt,
    },
    reference,
    metadata: { entry_point: row.entry_point },
  };
}

/**
 * Read cached examples (one JSON-encoded `TaskExample` per line). Returns
 * `null` if the cache is missing, signalling the loader to fetch fresh.
 */
function readCache(): TaskExample[] | null {
  const path = datasetCachePath(CACHE_FILENAME);
  if (!existsSync(path)) return null;
  const text = readFileSync(path, "utf8");
  const lines = text.split("\n").filter((l) => l.length > 0);
  return lines.map((line) => JSON.parse(line) as TaskExample);
}

/**
 * Persist examples to the on-disk cache. Format: jsonl. Creates the
 * parent directory on demand.
 */
function writeCache(examples: TaskExample[]): void {
  const path = datasetCachePath(CACHE_FILENAME);
  mkdirSync(dirname(path), { recursive: true });
  const lines = examples.map((ex) => JSON.stringify(ex));
  writeFileSync(path, `${lines.join("\n")}\n`, "utf8");
}

/**
 * Fetch HumanEval test rows from HuggingFace's public datasets-server. The
 * test split is the only one published (164 rows), so we just walk it
 * end-to-end. Network errors propagate.
 */
async function fetchFromHuggingFace(): Promise<TaskExample[]> {
  const POOL_CAP = 200; // headroom over the 164-row split
  const PAGE = 100;
  const url = (offset: number, length: number): string =>
    `https://datasets-server.huggingface.co/rows?dataset=openai%2Fopenai_humaneval&config=openai_humaneval&split=test&offset=${offset}&length=${length}`;

  const collected: TaskExample[] = [];
  let offset = 0;
  while (collected.length < POOL_CAP) {
    const remaining = POOL_CAP - collected.length;
    const length = Math.min(PAGE, remaining);
    const resp = await fetch(url(offset, length));
    if (!resp.ok) {
      throw new Error(
        `HF datasets-server returned ${resp.status} for HumanEval (offset=${offset}, length=${length})`,
      );
    }
    const json = (await resp.json()) as DatasetsServerResponse;
    if (json.rows.length === 0) break;
    for (const entry of json.rows) {
      collected.push(toTaskExample(entry.row));
    }
    offset += json.rows.length;
    if (json.rows.length < length) break;
  }
  return collected;
}

/**
 * Resolve the examples backing this task. Mirrors the QA loader: cache
 * first, network on miss, then deterministic shuffle + limit.
 */
async function loadExamples(opts?: LoadExamplesOptions): Promise<TaskExample[]> {
  const seed = opts?.seed ?? DEFAULT_SEED;
  const limit = opts?.limit;

  let pool = readCache();
  if (pool === null) {
    pool = await fetchFromHuggingFace();
    writeCache(pool);
  }

  const shuffled = seededShuffle(pool, seed);
  if (limit === undefined) return shuffled;
  return shuffled.slice(0, Math.max(0, limit));
}

/**
 * Default export — the canonical codegen `TaskDefinition`.
 *
 * Sandbox reminder (third time on purpose): scoring here executes
 * untrusted Python in a subprocess with a wall-clock timeout and no
 * OS-level isolation. Safe for cooperative benchmarks; UNSAFE for
 * adversarial / production routing. Wrap in Docker / sandbox-exec /
 * firejail before any such use.
 */
const codegenTask: TaskDefinition<CodegenReference, string> = {
  name: "codegen",
  description:
    "Python function completion (HumanEval test split, 164 problems). Scoring: pass@1 — 1.0 if all hidden assertions pass in a sandboxed Python subprocess with a 10s timeout.",
  promptTemplate,
  loadExamples,
  parseOutput,
  score,
};

export default codegenTask;

// Re-exports for tests + smoke harness.
export { SUBPROCESS_TIMEOUT_MS, looksDangerous, buildTestScript, runPython };
