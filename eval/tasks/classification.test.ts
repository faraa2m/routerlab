// eval/tasks/classification.test.ts — unit tests for sentiment classification.
//
// Run under `bun test`. Pure-CPU, no network. The dataset cache is seeded
// from an in-memory fixture so HF is never contacted.

import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { dirname } from "node:path";
import classificationTask, {
  parseOutput,
  promptTemplate,
  score,
  type SentimentLabel,
} from "./classification.ts";
import { datasetCachePath, type TaskExample } from "./_types.ts";

/**
 * Six-example fixture pool covering all three labels twice.
 */
const FIXTURE_POOL: TaskExample[] = [
  {
    id: "cls_test_1",
    input: { text: "I love this new phone!" },
    reference: "positive" satisfies SentimentLabel,
  },
  {
    id: "cls_test_2",
    input: { text: "I hate when it rains all day." },
    reference: "negative" satisfies SentimentLabel,
  },
  {
    id: "cls_test_3",
    input: { text: "The meeting is at 3pm." },
    reference: "neutral" satisfies SentimentLabel,
  },
  {
    id: "cls_test_4",
    input: { text: "Best concert ever!" },
    reference: "positive" satisfies SentimentLabel,
  },
  {
    id: "cls_test_5",
    input: { text: "Worst customer service experience of my life." },
    reference: "negative" satisfies SentimentLabel,
  },
  {
    id: "cls_test_6",
    input: { text: "Tomorrow is Tuesday." },
    reference: "neutral" satisfies SentimentLabel,
  },
];

const CACHE_PATH = datasetCachePath("classification.jsonl");
const BACKUP_PATH = `${CACHE_PATH}.testbak`;

/**
 * Seed the on-disk cache with the in-memory fixture. Backs up any existing
 * cache so a developer's live dataset survives the test run.
 */
function seedCache(): void {
  mkdirSync(dirname(CACHE_PATH), { recursive: true });
  if (existsSync(CACHE_PATH)) {
    copyFileSync(CACHE_PATH, BACKUP_PATH);
  }
  const lines = FIXTURE_POOL.map((ex) => JSON.stringify(ex));
  writeFileSync(CACHE_PATH, `${lines.join("\n")}\n`, "utf8");
}

/**
 * Restore the original cache (or remove the fixture if there was none).
 */
function restoreCache(): void {
  if (existsSync(BACKUP_PATH)) {
    copyFileSync(BACKUP_PATH, CACHE_PATH);
    rmSync(BACKUP_PATH);
  } else if (existsSync(CACHE_PATH)) {
    rmSync(CACHE_PATH);
  }
}

beforeAll(() => {
  seedCache();
});

afterAll(() => {
  restoreCache();
});

describe("classification promptTemplate", () => {
  test("contains the text and the label menu", () => {
    const prompt = promptTemplate({ text: "great day!" });
    expect(prompt).toContain("Tweet: great day!");
    expect(prompt).toContain("negative");
    expect(prompt).toContain("neutral");
    expect(prompt).toContain("positive");
    expect(prompt.length).toBeGreaterThan(0);
    expect(prompt.trim().endsWith("Sentiment:")).toBe(true);
  });

  test("handles empty / missing input", () => {
    const prompt = promptTemplate({});
    expect(prompt.length).toBeGreaterThan(0);
    expect(prompt).toContain("Tweet:");
  });
});

describe("classification parseOutput", () => {
  test("parses canonical labels", () => {
    expect(parseOutput("positive")).toBe("positive");
    expect(parseOutput("negative")).toBe("negative");
    expect(parseOutput("neutral")).toBe("neutral");
  });

  test("is case-insensitive", () => {
    expect(parseOutput("POSITIVE")).toBe("positive");
    expect(parseOutput("Positive")).toBe("positive");
  });

  test("trims and takes first word", () => {
    expect(parseOutput("  positive  ")).toBe("positive");
    expect(parseOutput("positive sentiment")).toBe("positive");
  });

  test("handles multi-line: first line wins", () => {
    expect(parseOutput("positive\nactually I'm not sure")).toBe("positive");
  });

  test("accepts short aliases", () => {
    expect(parseOutput("pos")).toBe("positive");
    expect(parseOutput("neg")).toBe("negative");
    expect(parseOutput("neu")).toBe("neutral");
  });

  test("accepts +/- symbols", () => {
    expect(parseOutput("+")).toBe("positive");
    expect(parseOutput("-")).toBe("negative");
  });

  test("returns null for empty / unparseable input", () => {
    expect(parseOutput("")).toBeNull();
    expect(parseOutput("   ")).toBeNull();
    expect(parseOutput("hello world")).toBeNull();
    expect(parseOutput("???")).toBeNull();
  });

  test("scans tokens after a leading garbage word", () => {
    // Some models prefix with "Sentiment:" or similar — first matchable token wins.
    expect(parseOutput("sentiment positive")).toBe("positive");
    expect(parseOutput("answer: neg")).toBe("negative");
  });
});

describe("classification score", () => {
  test("1.0 on exact match", () => {
    expect(score("positive", "positive")).toBe(1);
    expect(score("negative", "negative")).toBe(1);
    expect(score("neutral", "neutral")).toBe(1);
  });

  test("0.0 on mismatch", () => {
    expect(score("positive", "negative")).toBe(0);
    expect(score("neutral", "positive")).toBe(0);
  });

  test("0.0 on unparseable prediction (null)", () => {
    expect(score(null, "positive")).toBe(0);
    expect(score(null, "neutral")).toBe(0);
  });

  test("always returns 0 or 1", () => {
    const labels: SentimentLabel[] = ["negative", "neutral", "positive"];
    for (const pred of labels) {
      for (const ref of labels) {
        const s = score(pred, ref);
        expect(s === 0 || s === 1).toBe(true);
      }
    }
  });
});

describe("classification loadExamples", () => {
  test("returns the requested number of examples", async () => {
    const examples = await classificationTask.loadExamples({
      limit: 3,
      seed: 42,
    });
    expect(examples).toHaveLength(3);
  });

  test("each example has the expected shape", async () => {
    const examples = await classificationTask.loadExamples({
      limit: 3,
      seed: 42,
    });
    for (const ex of examples) {
      expect(typeof ex.id).toBe("string");
      expect(ex.id.length).toBeGreaterThan(0);
      expect(typeof ex.input.text).toBe("string");
      const ref = ex.reference as SentimentLabel;
      expect(["negative", "neutral", "positive"]).toContain(ref);
    }
  });

  test("seed determines order", async () => {
    const a = await classificationTask.loadExamples({ limit: 6, seed: 1 });
    const b = await classificationTask.loadExamples({ limit: 6, seed: 1 });
    const c = await classificationTask.loadExamples({ limit: 6, seed: 2 });
    expect(a.map((e) => e.id)).toEqual(b.map((e) => e.id));
    expect(a.map((e) => e.id)).not.toEqual(c.map((e) => e.id));
  });

  test("limit 0 returns empty array", async () => {
    const examples = await classificationTask.loadExamples({
      limit: 0,
      seed: 42,
    });
    expect(examples).toHaveLength(0);
  });
});

describe("classification default export", () => {
  test("exposes expected TaskDefinition shape", () => {
    expect(classificationTask.name).toBe("classification");
    expect(typeof classificationTask.description).toBe("string");
    expect(typeof classificationTask.promptTemplate).toBe("function");
    expect(typeof classificationTask.loadExamples).toBe("function");
    expect(typeof classificationTask.parseOutput).toBe("function");
    expect(typeof classificationTask.score).toBe("function");
  });
});
