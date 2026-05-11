// eval/tasks/qa.test.ts — unit tests for the QA task definition.
//
// These run under `bun test`. They are pure-CPU: no network, no disk
// reads outside of writing to a temp cache that the test seeds itself.
// The cache is seeded with a hand-crafted fixture so we never call HF
// during tests.

import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { dirname } from "node:path";
import qaTask, {
  isAbstain,
  normalizeAnswer,
  parseOutput,
  promptTemplate,
  score,
  tokenF1,
  type QaReference,
} from "./qa.ts";
import { datasetCachePath, type TaskExample } from "./_types.ts";

/**
 * Tiny in-memory fixture pool. Five examples — enough to exercise
 * limit / shuffle and both answerable + unanswerable rows.
 */
const FIXTURE_POOL: TaskExample[] = [
  {
    id: "qa_test_1",
    input: {
      context: "The Amazon River is the largest river by discharge volume in the world.",
      question: "What is the largest river by discharge volume?",
    },
    reference: { goldAnswers: ["The Amazon River", "Amazon River", "Amazon"], isImpossible: false } satisfies QaReference,
  },
  {
    id: "qa_test_2",
    input: {
      context: "Pluto was reclassified as a dwarf planet in 2006 by the IAU.",
      question: "When was Pluto reclassified?",
    },
    reference: { goldAnswers: ["2006", "in 2006"], isImpossible: false } satisfies QaReference,
  },
  {
    id: "qa_test_3",
    input: {
      context: "Mount Everest is the tallest mountain above sea level on Earth.",
      question: "Where on Mars is the tallest mountain?",
    },
    reference: { goldAnswers: [], isImpossible: true } satisfies QaReference,
  },
  {
    id: "qa_test_4",
    input: {
      context: "Python was created by Guido van Rossum and first released in 1991.",
      question: "Who created Python?",
    },
    reference: { goldAnswers: ["Guido van Rossum"], isImpossible: false } satisfies QaReference,
  },
  {
    id: "qa_test_5",
    input: {
      context: "The Eiffel Tower is in Paris, France.",
      question: "Where is the Statue of Liberty?",
    },
    reference: { goldAnswers: [], isImpossible: true } satisfies QaReference,
  },
];

const CACHE_PATH = datasetCachePath("qa.jsonl");
const BACKUP_PATH = `${CACHE_PATH}.testbak`;

/**
 * Seed the on-disk cache with the in-memory fixture so `loadExamples` is
 * offline-deterministic. Backs up any existing cache so we don't trample
 * a developer's live dataset.
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
 * Honors the "don't break the developer's existing dataset" rule.
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

describe("qa promptTemplate", () => {
  test("interpolates context and question", () => {
    const prompt = promptTemplate({
      context: "Hello world.",
      question: "What is the greeting?",
    });
    expect(prompt).toContain("Passage: Hello world.");
    expect(prompt).toContain("Question: What is the greeting?");
    expect(prompt.length).toBeGreaterThan(0);
    expect(prompt.trim().endsWith("Answer:")).toBe(true);
  });

  test("handles missing input keys without crashing", () => {
    const prompt = promptTemplate({});
    expect(prompt.length).toBeGreaterThan(0);
    expect(prompt).toContain("Passage:");
    expect(prompt).toContain("Question:");
  });
});

describe("qa parseOutput", () => {
  test("trims whitespace", () => {
    expect(parseOutput("  Paris  ")).toBe("Paris");
  });

  test("returns empty string for empty input", () => {
    expect(parseOutput("")).toBe("");
    expect(parseOutput("   ")).toBe("");
  });

  test("takes first line of multi-line output", () => {
    expect(parseOutput("Paris\nFrance\nCapital")).toBe("Paris");
    expect(parseOutput("Paris\r\nFrance")).toBe("Paris");
  });

  test("handles single line without trailing newline", () => {
    expect(parseOutput("Guido van Rossum")).toBe("Guido van Rossum");
  });
});

describe("qa normalizeAnswer", () => {
  test("lowercases", () => {
    expect(normalizeAnswer("Paris")).toBe("paris");
  });

  test("strips punctuation", () => {
    expect(normalizeAnswer("Paris, France!")).toBe("paris france");
  });

  test("drops articles a / an / the", () => {
    expect(normalizeAnswer("The Amazon")).toBe("amazon");
    expect(normalizeAnswer("a cat and an owl")).toBe("cat and owl");
  });

  test("collapses whitespace", () => {
    expect(normalizeAnswer("hello    world\n\nfoo")).toBe("hello world foo");
  });

  test("empty input yields empty string", () => {
    expect(normalizeAnswer("")).toBe("");
    expect(normalizeAnswer("   ")).toBe("");
  });
});

describe("qa tokenF1", () => {
  test("exact match → 1.0", () => {
    expect(tokenF1("Paris", "Paris")).toBe(1);
    expect(tokenF1("the eiffel tower", "Eiffel Tower")).toBe(1); // articles stripped both sides
  });

  test("no overlap → 0.0", () => {
    expect(tokenF1("Paris", "Berlin")).toBe(0);
  });

  test("partial overlap returns value in (0, 1)", () => {
    const f1 = tokenF1("Guido van Rossum", "van Rossum");
    expect(f1).toBeGreaterThan(0);
    expect(f1).toBeLessThan(1);
    // precision=2/3, recall=2/2=1, F1 = 2*(2/3)*1 / (2/3+1) = 0.8
    expect(f1).toBeCloseTo(0.8, 5);
  });

  test("both empty → 1.0", () => {
    expect(tokenF1("", "")).toBe(1);
  });

  test("only one empty → 0.0", () => {
    expect(tokenF1("Paris", "")).toBe(0);
    expect(tokenF1("", "Paris")).toBe(0);
  });

  test("repeated tokens are counted with multiplicity", () => {
    // pred: "Paris Paris" (2 toks), gold: "Paris" (1 tok)
    // common = 1 (we only get credit once), precision=1/2, recall=1, F1=2/3.
    expect(tokenF1("Paris Paris", "Paris")).toBeCloseTo(2 / 3, 5);
  });
});

describe("qa isAbstain", () => {
  test("recognizes common abstain phrasings", () => {
    expect(isAbstain("No answer")).toBe(true);
    expect(isAbstain("unanswerable")).toBe(true);
    expect(isAbstain("The passage does not say.")).toBe(true);
    expect(isAbstain("I don't know")).toBe(true);
    expect(isAbstain("")).toBe(true); // empty counts as abstain
  });

  test("does not flag substantive answers", () => {
    expect(isAbstain("Paris")).toBe(false);
    expect(isAbstain("The Amazon River")).toBe(false);
  });
});

describe("qa score", () => {
  test("answerable: 1.0 on exact match against any gold", () => {
    const ref: QaReference = {
      goldAnswers: ["Paris", "Paris, France"],
      isImpossible: false,
    };
    expect(score("Paris", ref)).toBe(1);
    expect(score("Paris, France", ref)).toBe(1);
  });

  test("answerable: takes max across golds", () => {
    const ref: QaReference = {
      goldAnswers: ["Berlin", "Paris"],
      isImpossible: false,
    };
    expect(score("Paris", ref)).toBe(1);
  });

  test("answerable: 0.0 with no overlap", () => {
    const ref: QaReference = { goldAnswers: ["Paris"], isImpossible: false };
    expect(score("Berlin", ref)).toBe(0);
  });

  test("unanswerable: 1.0 on abstain phrasing", () => {
    const ref: QaReference = { goldAnswers: [], isImpossible: true };
    expect(score("No answer", ref)).toBe(1);
    expect(score("unanswerable", ref)).toBe(1);
  });

  test("unanswerable: 0.0 on substantive guess", () => {
    const ref: QaReference = { goldAnswers: [], isImpossible: true };
    expect(score("Paris", ref)).toBe(0);
  });

  test("always returns a value in [0, 1]", () => {
    const cases: Array<[string, QaReference]> = [
      ["Paris", { goldAnswers: ["Paris"], isImpossible: false }],
      ["", { goldAnswers: ["Paris"], isImpossible: false }],
      ["Berlin", { goldAnswers: ["Paris"], isImpossible: false }],
      ["No answer", { goldAnswers: [], isImpossible: true }],
      ["Paris", { goldAnswers: [], isImpossible: true }],
    ];
    for (const [pred, ref] of cases) {
      const s = score(pred, ref);
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });
});

describe("qa loadExamples", () => {
  test("returns the requested number of examples", async () => {
    const examples = await qaTask.loadExamples({ limit: 3, seed: 42 });
    expect(examples).toHaveLength(3);
  });

  test("each example has the expected shape", async () => {
    const examples = await qaTask.loadExamples({ limit: 3, seed: 42 });
    for (const ex of examples) {
      expect(typeof ex.id).toBe("string");
      expect(ex.id.length).toBeGreaterThan(0);
      expect(typeof ex.input.context).toBe("string");
      expect(typeof ex.input.question).toBe("string");
      const ref = ex.reference as QaReference;
      expect(Array.isArray(ref.goldAnswers)).toBe(true);
      expect(typeof ref.isImpossible).toBe("boolean");
    }
  });

  test("seed determines order", async () => {
    const a = await qaTask.loadExamples({ limit: 5, seed: 1 });
    const b = await qaTask.loadExamples({ limit: 5, seed: 1 });
    const c = await qaTask.loadExamples({ limit: 5, seed: 2 });
    expect(a.map((e) => e.id)).toEqual(b.map((e) => e.id));
    // Different seed → at least one order difference in our 5-item pool.
    expect(a.map((e) => e.id)).not.toEqual(c.map((e) => e.id));
  });

  test("limit 0 returns empty array", async () => {
    const examples = await qaTask.loadExamples({ limit: 0, seed: 42 });
    expect(examples).toHaveLength(0);
  });
});

describe("qa default export", () => {
  test("exposes expected TaskDefinition shape", () => {
    expect(qaTask.name).toBe("qa");
    expect(typeof qaTask.description).toBe("string");
    expect(typeof qaTask.promptTemplate).toBe("function");
    expect(typeof qaTask.loadExamples).toBe("function");
    expect(typeof qaTask.parseOutput).toBe("function");
    expect(typeof qaTask.score).toBe("function");
  });
});
