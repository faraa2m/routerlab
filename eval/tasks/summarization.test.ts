// summarization.test.ts — unit tests for the XSum summarization task.
//
// Network-free: we seed the on-disk cache from a bundled fixture so
// `loadExamples` never reaches HuggingFace.

import { describe, expect, test, beforeAll, afterAll } from "bun:test";
import { existsSync, mkdirSync, copyFileSync, rmSync } from "node:fs";
import { dirname } from "node:path";

import summarizationTask, {
  lcsLength,
  parseOutput,
  promptTemplate,
  rougeLF1,
  score,
  tokenize,
  type SummarizationReference,
} from "./summarization.ts";
import { datasetCachePath } from "./_types.ts";

const FIXTURE_PATH = new URL("./fixtures/summarization.jsonl", import.meta.url).pathname;

function seedCache(): void {
  const cachePath = datasetCachePath("summarization.jsonl");
  mkdirSync(dirname(cachePath), { recursive: true });
  copyFileSync(FIXTURE_PATH, cachePath);
}

beforeAll(() => {
  const cachePath = datasetCachePath("summarization.jsonl");
  if (existsSync(cachePath)) rmSync(cachePath);
  seedCache();
});

afterAll(() => {
  const cachePath = datasetCachePath("summarization.jsonl");
  if (existsSync(cachePath)) rmSync(cachePath);
});

describe("summarization — loadExamples", () => {
  test("returns 3 valid examples with limit=3", async () => {
    const examples = await summarizationTask.loadExamples({ limit: 3 });
    expect(examples.length).toBe(3);
    for (const ex of examples) {
      expect(typeof ex.id).toBe("string");
      const doc = ex.input.document ?? "";
      expect(typeof doc).toBe("string");
      expect(doc.length).toBeGreaterThan(0);
      const ref = ex.reference as SummarizationReference;
      expect(typeof ref.goldSummary).toBe("string");
      expect(ref.goldSummary.length).toBeGreaterThan(0);
    }
  });

  test("loadExamples is deterministic across calls", async () => {
    const a = await summarizationTask.loadExamples({ limit: 2, seed: 11 });
    const b = await summarizationTask.loadExamples({ limit: 2, seed: 11 });
    expect(a.map((e) => e.id)).toEqual(b.map((e) => e.id));
  });
});

describe("summarization — promptTemplate", () => {
  test("renders the prompt with the article and the single-sentence clause", () => {
    const rendered = promptTemplate({ document: "Some news article body." });
    expect(rendered).toContain("Summarize the following article in a single sentence.");
    expect(rendered).toContain("Article: Some news article body.");
    expect(rendered).toContain("Summary:");
  });

  test("handles empty input without throwing", () => {
    expect(() => promptTemplate({ document: "" })).not.toThrow();
  });
});

describe("summarization — parseOutput", () => {
  test("trims and takes the first non-empty line", () => {
    expect(parseOutput("  A summary sentence.  \nExtra commentary.")).toBe(
      "A summary sentence.",
    );
  });

  test("strips a leading 'Summary:' label", () => {
    expect(parseOutput("Summary: The summary text.")).toBe("The summary text.");
    expect(parseOutput("summary:  case insensitive")).toBe("case insensitive");
  });

  test("returns empty string for empty input", () => {
    expect(parseOutput("")).toBe("");
    expect(parseOutput("   \n\n")).toBe("");
  });

  test("returns the first non-empty line when leading blank lines are present", () => {
    expect(parseOutput("\n\n  Real summary.\n")).toBe("Real summary.");
  });
});

describe("summarization — tokenize", () => {
  test("lowercases and splits on whitespace + punctuation", () => {
    expect(tokenize("The quick, brown FOX!")).toEqual([
      "the",
      "quick",
      "brown",
      "fox",
    ]);
  });

  test("collapses runs of punctuation into one separator", () => {
    expect(tokenize("a... b!!! c")).toEqual(["a", "b", "c"]);
  });

  test("returns [] for empty / whitespace-only input", () => {
    expect(tokenize("")).toEqual([]);
    expect(tokenize("   ")).toEqual([]);
  });
});

describe("summarization — lcsLength", () => {
  test("computes LCS length correctly on canonical examples", () => {
    expect(lcsLength(["a", "b", "c"], ["a", "b", "c"])).toBe(3);
    expect(lcsLength(["a", "b", "c"], ["a", "c"])).toBe(2);
    expect(lcsLength(["a", "b", "c"], ["c", "b", "a"])).toBe(1);
    expect(lcsLength(["a", "b", "c"], ["x", "y", "z"])).toBe(0);
  });

  test("handles empty arrays", () => {
    expect(lcsLength([], ["a", "b"])).toBe(0);
    expect(lcsLength(["a"], [])).toBe(0);
    expect(lcsLength([], [])).toBe(0);
  });

  test("handles arrays of different lengths", () => {
    expect(
      lcsLength(
        ["the", "quick", "brown", "fox"],
        ["the", "lazy", "fox"],
      ),
    ).toBe(2);
  });
});

describe("summarization — rougeLF1", () => {
  test("returns 1.0 for identical strings", () => {
    expect(rougeLF1("The cat sat on the mat", "The cat sat on the mat")).toBe(1);
  });

  test("returns 0.0 for completely disjoint strings", () => {
    expect(rougeLF1("apple pear", "spaceship rocket")).toBe(0);
  });

  test("returns a value in (0, 1) for partial overlap", () => {
    const f1 = rougeLF1("the cat sat on the mat", "the cat sat down");
    expect(f1).toBeGreaterThan(0);
    expect(f1).toBeLessThan(1);
  });

  test("is order-sensitive (LCS, not token bag)", () => {
    // Same tokens, different order — LCS is shorter.
    const f1Aligned = rougeLF1("a b c d", "a b c d");
    const f1Reversed = rougeLF1("a b c d", "d c b a");
    expect(f1Aligned).toBeGreaterThan(f1Reversed);
  });

  test("returns 1.0 for two empty strings (vacuous match)", () => {
    expect(rougeLF1("", "")).toBe(1);
  });

  test("returns 0.0 when one side is empty", () => {
    expect(rougeLF1("the cat", "")).toBe(0);
    expect(rougeLF1("", "the cat")).toBe(0);
  });

  test("F1 formula sanity: known LCS 2 over 3 / 3 tokens yields ~0.6667", () => {
    // "the cat sat" vs "the cat ran": LCS = 2 (the, cat). P = 2/3, R = 2/3, F1 = 2/3.
    const f1 = rougeLF1("the cat sat", "the cat ran");
    expect(f1).toBeCloseTo(2 / 3, 5);
  });
});

describe("summarization — score", () => {
  test("delegates to rougeLF1 on the gold summary", () => {
    const ref: SummarizationReference = { goldSummary: "The cat sat on the mat." };
    const result = score("The cat sat on the mat.", ref);
    expect(result).toBe(1);
  });

  test("returns 0 for completely wrong summary", () => {
    const ref: SummarizationReference = { goldSummary: "Liverpool beat City." };
    expect(score("Apples and pears", ref)).toBe(0);
  });
});

describe("summarization — task definition shape", () => {
  test("exports a TaskDefinition with the canonical name", () => {
    expect(summarizationTask.name).toBe("summarization");
    expect(typeof summarizationTask.description).toBe("string");
    expect(typeof summarizationTask.promptTemplate).toBe("function");
    expect(typeof summarizationTask.loadExamples).toBe("function");
    expect(typeof summarizationTask.parseOutput).toBe("function");
    expect(typeof summarizationTask.score).toBe("function");
  });
});
