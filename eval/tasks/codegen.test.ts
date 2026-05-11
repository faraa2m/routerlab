// codegen.test.ts — unit tests for the HumanEval task definition.
//
// These tests are offline by construction: we seed the on-disk cache from
// a hand-crafted fixture so `loadExamples` does NOT hit the network.
// The cache lookup runs *first* inside the task module, so seeding it
// up front gives us a fully deterministic run.
//
// Sandbox-sensitive tests (the ones that spawn `python3`) are gated on
// the binary being present. When `python3` is missing on the test host,
// those tests skip rather than fail — keeps CI green on minimal images
// while still exercising the sandbox locally.

import { describe, expect, test, beforeAll, afterAll } from "bun:test";
import { existsSync, mkdirSync, copyFileSync, rmSync } from "node:fs";
import { dirname } from "node:path";
import { spawnSync } from "node:child_process";

import codegenTask, {
  buildTestScript,
  looksDangerous,
  parseOutput,
  promptTemplate,
  runPython,
  score,
  SUBPROCESS_TIMEOUT_MS,
  type CodegenReference,
} from "./codegen.ts";
import { datasetCachePath } from "./_types.ts";

const FIXTURE_PATH = new URL("./fixtures/codegen.jsonl", import.meta.url).pathname;

/**
 * Seed the on-disk cache from the bundled fixture before tests run.
 * Idempotent: re-running just rewrites the file.
 */
function seedCache(): void {
  const cachePath = datasetCachePath("codegen.jsonl");
  mkdirSync(dirname(cachePath), { recursive: true });
  copyFileSync(FIXTURE_PATH, cachePath);
}

/**
 * Probe for `python3` once. Tests that need it skip when it's not present.
 */
function pythonAvailable(): boolean {
  try {
    const r = spawnSync("python3", ["--version"], { stdio: "ignore" });
    return r.status === 0;
  } catch {
    return false;
  }
}

const HAVE_PYTHON = pythonAvailable();

beforeAll(() => {
  // Wipe any previously-cached file from a real network fetch so the
  // fixture is the source of truth for these tests.
  const cachePath = datasetCachePath("codegen.jsonl");
  if (existsSync(cachePath)) rmSync(cachePath);
  seedCache();
});

afterAll(() => {
  // Clean up so subsequent runs of `_smoke.ts` re-fetch a real HF slice.
  const cachePath = datasetCachePath("codegen.jsonl");
  if (existsSync(cachePath)) rmSync(cachePath);
});

describe("codegen — loadExamples", () => {
  test("returns 3 valid examples with limit=3", async () => {
    const examples = await codegenTask.loadExamples({ limit: 3 });
    expect(examples.length).toBe(3);
    for (const ex of examples) {
      expect(typeof ex.id).toBe("string");
      expect(ex.id.length).toBeGreaterThan(0);
      expect(typeof ex.input.prompt).toBe("string");
      const ref = ex.reference as CodegenReference;
      expect(typeof ref.prompt).toBe("string");
      expect(typeof ref.entryPoint).toBe("string");
      expect(typeof ref.testProgram).toBe("string");
      expect(ref.entryPoint.length).toBeGreaterThan(0);
    }
  });

  test("loadExamples is deterministic across calls with the same seed", async () => {
    const a = await codegenTask.loadExamples({ limit: 3, seed: 7 });
    const b = await codegenTask.loadExamples({ limit: 3, seed: 7 });
    expect(a.map((e) => e.id)).toEqual(b.map((e) => e.id));
  });
});

describe("codegen — promptTemplate", () => {
  test("renders the prompt with the function stub and the no-markdown clause", () => {
    const rendered = promptTemplate({ prompt: "def f(x):\n    pass\n" });
    expect(rendered).toContain("Complete the following Python function");
    expect(rendered).toContain("Output only the function body");
    expect(rendered).toContain("def f(x):");
  });

  test("handles an empty prompt without throwing", () => {
    expect(() => promptTemplate({ prompt: "" })).not.toThrow();
  });
});

describe("codegen — parseOutput", () => {
  test("strips a fenced ```python block", () => {
    const raw = "```python\n    return a + b\n```";
    expect(parseOutput(raw)).toBe("    return a + b");
  });

  test("strips a fenced block with no language tag", () => {
    const raw = "```\n    return 1\n```";
    expect(parseOutput(raw)).toBe("    return 1");
  });

  test("returns raw text when no fences are present", () => {
    const raw = "    return a + b";
    expect(parseOutput(raw)).toBe("    return a + b");
  });

  test("returns empty string on empty input", () => {
    expect(parseOutput("")).toBe("");
    expect(parseOutput("   \n  \n")).toBe("");
  });

  test("preserves indentation across multi-line completions", () => {
    const raw = "```python\n    if a > b:\n        return a\n    return b\n```";
    const parsed = parseOutput(raw);
    expect(parsed).toContain("    if a > b:");
    expect(parsed).toContain("        return a");
  });
});

describe("codegen — looksDangerous", () => {
  test("flags os/subprocess/socket imports", () => {
    expect(looksDangerous("import os\nreturn 1")).toBe(true);
    expect(looksDangerous("import subprocess; subprocess.run(['ls'])")).toBe(true);
    expect(looksDangerous("import socket")).toBe(true);
  });

  test("flags eval/exec/__import__/open", () => {
    expect(looksDangerous("eval('1+1')")).toBe(true);
    expect(looksDangerous("exec('print(1)')")).toBe(true);
    expect(looksDangerous("__import__('os')")).toBe(true);
    expect(looksDangerous("open('/etc/passwd')")).toBe(true);
  });

  test("does not flag innocuous completions", () => {
    expect(looksDangerous("    return a + b")).toBe(false);
    expect(looksDangerous("    if x > 0:\n        return x\n    return 0")).toBe(false);
    expect(looksDangerous("    return sum(xs) / len(xs)")).toBe(false);
  });
});

describe("codegen — buildTestScript", () => {
  test("composes prompt header, completion, test program, and check call", () => {
    const ref: CodegenReference = {
      prompt: "def add(a, b):\n",
      entryPoint: "add",
      testProgram: "def check(candidate):\n    assert candidate(1, 2) == 3\n",
    };
    const script = buildTestScript("    return a + b", ref);
    expect(script).toContain("def add(a, b):");
    expect(script).toContain("    return a + b");
    expect(script).toContain("def check(candidate):");
    expect(script).toContain("check(add)");
  });
});

describe("codegen — score (sandboxed Python execution)", () => {
  const ref: CodegenReference = {
    prompt: "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n",
    entryPoint: "add",
    testProgram:
      "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n    assert candidate(-1, 1) == 0\n",
  };

  test.skipIf(!HAVE_PYTHON)("scores 1.0 on a correct completion", async () => {
    const result = await score("    return a + b", ref);
    expect(result).toBe(1);
  });

  test.skipIf(!HAVE_PYTHON)("scores 0.0 on a deliberately broken completion", async () => {
    // Off by one: should fail the (1, 2) == 3 assertion.
    const result = await score("    return a + b + 1", ref);
    expect(result).toBe(0);
  });

  test.skipIf(!HAVE_PYTHON)("scores 0.0 on a syntactically invalid completion", async () => {
    // Missing indentation under `def`: Python raises an IndentationError.
    const result = await score("return a + b", ref);
    expect(result).toBe(0);
  });

  test("scores 0.0 on an empty completion without spawning", async () => {
    const result = await score("", ref);
    expect(result).toBe(0);
  });

  test("scores 0.0 on a dangerous-looking completion without spawning", async () => {
    const result = await score("    import os; return os.getcwd()", ref);
    expect(result).toBe(0);
  });
});

describe("codegen — runPython (sandbox primitive)", () => {
  test.skipIf(!HAVE_PYTHON)("returns ok=true for a trivial program", async () => {
    const r = await runPython("print('hi')", SUBPROCESS_TIMEOUT_MS);
    expect(r.ok).toBe(true);
  });

  test.skipIf(!HAVE_PYTHON)("returns ok=false for a program that raises", async () => {
    const r = await runPython("raise SystemExit(1)", SUBPROCESS_TIMEOUT_MS);
    expect(r.ok).toBe(false);
  });

  test.skipIf(!HAVE_PYTHON)("kills a program that exceeds the timeout", async () => {
    // 500ms timeout, infinite loop body.
    const r = await runPython("while True: pass", 500);
    expect(r.ok).toBe(false);
    expect(r.stderr).toContain("[timeout]");
  });
});

describe("codegen — task definition shape", () => {
  test("exports a TaskDefinition with the canonical name", () => {
    expect(codegenTask.name).toBe("codegen");
    expect(typeof codegenTask.description).toBe("string");
    expect(typeof codegenTask.promptTemplate).toBe("function");
    expect(typeof codegenTask.loadExamples).toBe("function");
    expect(typeof codegenTask.parseOutput).toBe("function");
    expect(typeof codegenTask.score).toBe("function");
  });
});
