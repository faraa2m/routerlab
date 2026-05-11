// reasoning.test.ts — unit tests for the GSM8K reasoning task.
//
// Network-free: we seed the on-disk cache from a bundled fixture so
// `loadExamples` never reaches HuggingFace.

import { describe, expect, test, beforeAll, afterAll } from "bun:test";
import { existsSync, mkdirSync, copyFileSync, rmSync } from "node:fs";
import { dirname } from "node:path";

import reasoningTask, {
  extractReferenceAnswer,
  normalizeNumber,
  parseOutput,
  promptTemplate,
  score,
  type ReasoningReference,
} from "./reasoning.ts";
import { datasetCachePath } from "./_types.ts";

const FIXTURE_PATH = new URL("./fixtures/reasoning.jsonl", import.meta.url).pathname;

function seedCache(): void {
  const cachePath = datasetCachePath("reasoning.jsonl");
  mkdirSync(dirname(cachePath), { recursive: true });
  copyFileSync(FIXTURE_PATH, cachePath);
}

beforeAll(() => {
  const cachePath = datasetCachePath("reasoning.jsonl");
  if (existsSync(cachePath)) rmSync(cachePath);
  seedCache();
});

afterAll(() => {
  const cachePath = datasetCachePath("reasoning.jsonl");
  if (existsSync(cachePath)) rmSync(cachePath);
});

describe("reasoning — loadExamples", () => {
  test("returns 3 valid examples with limit=3", async () => {
    const examples = await reasoningTask.loadExamples({ limit: 3 });
    expect(examples.length).toBe(3);
    for (const ex of examples) {
      expect(typeof ex.id).toBe("string");
      const question = ex.input.question ?? "";
      expect(typeof question).toBe("string");
      expect(question.length).toBeGreaterThan(0);
      const ref = ex.reference as ReasoningReference;
      expect(typeof ref.goldAnswer).toBe("string");
      expect(ref.goldAnswer.length).toBeGreaterThan(0);
    }
  });

  test("loadExamples is deterministic across calls", async () => {
    const a = await reasoningTask.loadExamples({ limit: 2, seed: 5 });
    const b = await reasoningTask.loadExamples({ limit: 2, seed: 5 });
    expect(a.map((e) => e.id)).toEqual(b.map((e) => e.id));
  });
});

describe("reasoning — promptTemplate", () => {
  test("renders the prompt with the question and the #### convention", () => {
    const rendered = promptTemplate({ question: "What is 2+2?" });
    expect(rendered).toContain("Solve this problem step by step.");
    expect(rendered).toContain("'####'");
    expect(rendered).toContain("Problem: What is 2+2?");
  });

  test("handles an empty question without throwing", () => {
    expect(() => promptTemplate({ question: "" })).not.toThrow();
  });
});

describe("reasoning — normalizeNumber", () => {
  test("strips commas and currency symbols", () => {
    expect(normalizeNumber("$1,234")).toBe("1234");
    expect(normalizeNumber("€999")).toBe("999");
  });

  test("normalizes trailing zeros in decimals", () => {
    expect(normalizeNumber("42.50")).toBe("42.5");
    expect(normalizeNumber("3.000")).toBe("3");
  });

  test("preserves integer form when input is integer", () => {
    expect(normalizeNumber("42")).toBe("42");
    expect(normalizeNumber("007")).toBe("7");
  });

  test("handles negative numbers", () => {
    expect(normalizeNumber("-5")).toBe("-5");
    expect(normalizeNumber("-1,200")).toBe("-1200");
  });

  test("returns empty string on non-numeric input", () => {
    expect(normalizeNumber("")).toBe("");
    expect(normalizeNumber("not a number")).toBe("");
  });
});

describe("reasoning — parseOutput", () => {
  test("extracts a number after ####", () => {
    expect(parseOutput("Some reasoning here.\n#### 42")).toBe("42");
  });

  test("handles #### with surrounding whitespace", () => {
    expect(parseOutput("text  ####   17  \nextra")).toBe("17");
  });

  test("falls back to the LAST numeric token when #### is absent", () => {
    expect(parseOutput("First I had 3 apples, then 5 more, so 8 total.")).toBe(
      "8",
    );
  });

  test("returns empty string when no number is present", () => {
    expect(parseOutput("")).toBe("");
    expect(parseOutput("I cannot answer this question.")).toBe("");
  });

  test("strips commas from extracted numbers", () => {
    expect(parseOutput("#### 1,234")).toBe("1234");
  });

  test("handles multi-line completions with #### at the end", () => {
    const raw = "Step 1: ...\nStep 2: ...\nFinal answer.\n#### 99";
    expect(parseOutput(raw)).toBe("99");
  });

  test("handles negative answers after ####", () => {
    expect(parseOutput("#### -3")).toBe("-3");
  });

  test("uses the LAST #### when multiple are present", () => {
    expect(parseOutput("Maybe #### 1 or perhaps #### 7")).toBe("7");
  });

  test("falls back to last numeric token when #### line has no number", () => {
    // #### line is empty; last number in body is 12.
    expect(parseOutput("answer is 12 ####")).toBe("12");
  });
});

describe("reasoning — extractReferenceAnswer", () => {
  test("extracts the final number from a GSM8K-style answer", () => {
    expect(
      extractReferenceAnswer(
        "Janet has 3 apples and buys 5, so 3 + 5 = 8.\n#### 8",
      ),
    ).toBe("8");
  });

  test("falls back to last numeric token when #### is missing", () => {
    expect(extractReferenceAnswer("She has 4 apples now")).toBe("4");
  });

  test("returns empty string on input with no numbers", () => {
    expect(extractReferenceAnswer("no numbers here")).toBe("");
  });
});

describe("reasoning — score", () => {
  test("returns 1.0 on exact match", () => {
    const ref: ReasoningReference = { goldAnswer: "42" };
    expect(score("42", ref)).toBe(1);
  });

  test("returns 0.0 on mismatch", () => {
    const ref: ReasoningReference = { goldAnswer: "42" };
    expect(score("43", ref)).toBe(0);
  });

  test("returns 0.0 on empty prediction", () => {
    const ref: ReasoningReference = { goldAnswer: "42" };
    expect(score("", ref)).toBe(0);
  });

  test("end-to-end: parseOutput → score on a chain-of-thought answer", () => {
    const ref: ReasoningReference = { goldAnswer: "8" };
    const raw = "Step 1: 3 + 5 = 8.\nStep 2: confirm.\n#### 8";
    expect(score(parseOutput(raw), ref)).toBe(1);
  });

  test("end-to-end: parseOutput → score on a wrong answer", () => {
    const ref: ReasoningReference = { goldAnswer: "8" };
    const raw = "I think it's 7.\n#### 7";
    expect(score(parseOutput(raw), ref)).toBe(0);
  });
});

describe("reasoning — task definition shape", () => {
  test("exports a TaskDefinition with the canonical name", () => {
    expect(reasoningTask.name).toBe("reasoning");
    expect(typeof reasoningTask.description).toBe("string");
    expect(typeof reasoningTask.promptTemplate).toBe("function");
    expect(typeof reasoningTask.loadExamples).toBe("function");
    expect(typeof reasoningTask.parseOutput).toBe("function");
    expect(typeof reasoningTask.score).toBe("function");
  });
});
