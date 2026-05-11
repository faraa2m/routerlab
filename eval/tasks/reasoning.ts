// eval/tasks/reasoning.ts — multi-step reasoning task definition for routerlab.
//
// Dataset: GSM8K (Cobbe et al., 2021, "Training Verifiers to Solve Math
// Word Problems").
//   Source:  https://huggingface.co/datasets/openai/gsm8k
//   Split:   test (1,319 examples).
//   License: MIT — redistribution-compatible with no notable restrictions.
//   Notes:   Each row is a grade-school math word problem with a chain-of-
//            thought rationale ending in `#### N` where N is the integer
//            final answer. We score against N only.
//
// Scoring: exact match on the numeric answer (1.0 / 0.0). The model is
// asked to follow the same `####` convention so the parser is symmetric
// across prediction and reference.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import {
  datasetCachePath,
  seededShuffle,
  type LoadExamplesOptions,
  type TaskDefinition,
  type TaskExample,
} from "./_types.ts";

const CACHE_FILENAME = "reasoning.jsonl";
const DEFAULT_SEED = 42;

/**
 * GSM8K reference payload. We carry only the final numeric answer; the
 * chain-of-thought rationale is dropped because we score on the answer
 * alone. The number is stored as a `string` so we don't lose precision
 * on the (rare) non-integer answers and so JSON round-trips cleanly.
 */
export interface ReasoningReference {
  /** The canonical answer string after `####`, normalized via `normalizeNumber`. */
  goldAnswer: string;
}

/**
 * HF datasets-server returns GSM8K rows in this shape.
 */
interface Gsm8kRow {
  question: string;
  answer: string;
}

interface DatasetsServerResponse {
  rows: Array<{ row: Gsm8kRow }>;
}

/**
 * Render the reasoning prompt. Pure: depends only on `input`. We instruct
 * the model to follow GSM8K's own convention (chain-of-thought followed
 * by `#### <number>`). This makes parsing symmetric and gives smaller
 * models a structural cue they often need to land the answer in a parseable
 * spot.
 */
export function promptTemplate(input: TaskExample["input"]): string {
  const question = input.question ?? "";
  return `Solve this problem step by step. Show your reasoning, then give the final numeric answer after '####'.

Problem: ${question}

Answer:`;
}

/**
 * Normalize a numeric string into a canonical form for equality comparison.
 *   - Strip commas and currency symbols ($ £ €).
 *   - Strip surrounding whitespace.
 *   - Convert `1,234.50` → `1234.5`; `42.0` → `42`; `+7` → `7`.
 *   - Preserve signs and decimals where the underlying value differs.
 *
 * Returns the canonical string. Non-numeric input returns the empty string
 * — equality against the empty string is `false` for any valid answer.
 */
export function normalizeNumber(s: string): string {
  if (s.length === 0) return "";
  // Strip currency and grouping commas first; keep digits, sign, decimal.
  const cleaned = s.replace(/[\$£€,_\s]/g, "");
  // Match an optional sign, digits, optional decimal portion.
  const m = cleaned.match(/-?\d+(\.\d+)?/);
  if (m === null) return "";
  const raw = m[0];
  // Normalize numeric form via Number → string round-trip. This drops
  // trailing zeros after a decimal (42.50 → 42.5) and unifies leading-
  // zero variants (007 → 7). For huge integers, Number can lose precision;
  // fall back to the raw match in that case.
  const n = Number(raw);
  if (!Number.isFinite(n)) return raw;
  const roundTripped = n.toString();
  // If the original looks like a plain integer, prefer the integer form.
  if (!raw.includes(".") && Number.isInteger(n)) {
    return roundTripped;
  }
  return roundTripped;
}

/**
 * Extract the numeric answer from a raw model completion.
 *   1. If the string contains `####`, take everything after the last
 *      occurrence and normalize it.
 *   2. Otherwise fall back to the *last* numeric token in the string.
 *
 * Returns the normalized number string, or "" if none was found.
 */
export function parseOutput(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.length === 0) return "";

  const hashIdx = trimmed.lastIndexOf("####");
  if (hashIdx !== -1) {
    const tail = trimmed.slice(hashIdx + 4);
    // Take only the first line after #### — models sometimes append a
    // signature or rationale recap. Normalize from the first numeric
    // token in that line.
    const firstLine = tail.split(/\r?\n/)[0] ?? "";
    const match = firstLine.match(/-?[\d,]+(\.\d+)?/);
    if (match !== null) {
      return normalizeNumber(match[0]);
    }
    // No number on the #### line — fall through to the last-token rule.
  }

  // Fallback: last numeric-looking token anywhere in the response. The
  // regex matches integers with optional grouping commas and optional
  // decimal portion. We pick the *last* match because models often think
  // out loud and the final number is the answer.
  const allMatches = trimmed.match(/-?[\d,]+(\.\d+)?/g);
  if (allMatches === null || allMatches.length === 0) return "";
  return normalizeNumber(allMatches[allMatches.length - 1] ?? "");
}

/**
 * Extract the gold answer from a GSM8K reference row. GSM8K rows end with
 * `\n#### <answer>` — we split on that marker and normalize. If the
 * marker is somehow missing (corrupt row), we fall back to the same
 * last-token rule used in `parseOutput` so a single bad row doesn't
 * crash the whole pipeline.
 */
export function extractReferenceAnswer(rawAnswer: string): string {
  const idx = rawAnswer.lastIndexOf("####");
  if (idx !== -1) {
    const tail = rawAnswer.slice(idx + 4).trim();
    const match = tail.match(/-?[\d,]+(\.\d+)?/);
    if (match !== null) return normalizeNumber(match[0]);
  }
  const matches = rawAnswer.match(/-?[\d,]+(\.\d+)?/g);
  if (matches === null || matches.length === 0) return "";
  return normalizeNumber(matches[matches.length - 1] ?? "");
}

/**
 * Score a parsed prediction against a GSM8K reference. Exact match on
 * the normalized numeric form: 1.0 if equal, else 0.0.
 *
 * We deliberately do NOT accept partial credit. GSM8K answers are short
 * integers and the goal is to measure end-to-end correctness, not how
 * close the model got. Partial credit on math is a tar pit (cf. the
 * "you almost have 42" memes); exact match is the standard.
 */
export function score(parsed: string, reference: ReasoningReference): number {
  if (parsed.length === 0) return 0;
  return parsed === reference.goldAnswer ? 1 : 0;
}

/**
 * Convert a GSM8K row into our `TaskExample` shape. The HF dataset
 * stores the chain-of-thought rationale inline in `answer`; we extract
 * the final number and discard the rationale (we only score on the
 * answer).
 */
function toTaskExample(row: Gsm8kRow, index: number): TaskExample {
  const reference: ReasoningReference = {
    goldAnswer: extractReferenceAnswer(row.answer),
  };
  return {
    // GSM8K rows don't carry stable ids; we synthesize one from the index
    // so the eval cache keys remain consistent across runs.
    id: `gsm8k-test-${index}`,
    input: { question: row.question },
    reference,
    metadata: { rawAnswer: row.answer },
  };
}

function readCache(): TaskExample[] | null {
  const path = datasetCachePath(CACHE_FILENAME);
  if (!existsSync(path)) return null;
  const text = readFileSync(path, "utf8");
  const lines = text.split("\n").filter((l) => l.length > 0);
  return lines.map((line) => JSON.parse(line) as TaskExample);
}

function writeCache(examples: TaskExample[]): void {
  const path = datasetCachePath(CACHE_FILENAME);
  mkdirSync(dirname(path), { recursive: true });
  const lines = examples.map((ex) => JSON.stringify(ex));
  writeFileSync(path, `${lines.join("\n")}\n`, "utf8");
}

/**
 * Fetch GSM8K test rows from HF datasets-server. GSM8K has two configs:
 * `main` (1,319 problems) and `socratic` (with subquestion prompts). We
 * use `main` for routerlab — it's the standard reasoning benchmark.
 */
async function fetchFromHuggingFace(cap: number): Promise<TaskExample[]> {
  const url = (offset: number, length: number): string =>
    `https://datasets-server.huggingface.co/rows?dataset=openai%2Fgsm8k&config=main&split=test&offset=${offset}&length=${length}`;
  const PAGE = 100;
  const collected: TaskExample[] = [];
  let offset = 0;
  while (collected.length < cap) {
    const remaining = cap - collected.length;
    const length = Math.min(PAGE, remaining);
    const resp = await fetch(url(offset, length));
    if (!resp.ok) {
      throw new Error(
        `HF datasets-server returned ${resp.status} for GSM8K (offset=${offset}, length=${length})`,
      );
    }
    const json = (await resp.json()) as DatasetsServerResponse;
    if (json.rows.length === 0) break;
    for (const entry of json.rows) {
      collected.push(toTaskExample(entry.row, offset + collected.length));
    }
    offset += json.rows.length;
    if (json.rows.length < length) break;
  }
  return collected;
}

/**
 * Resolve the examples backing this task. Pool capped at 1,000 — the
 * full GSM8K test split is 1,319 problems, so this is a near-complete
 * sample with deterministic shuffling for sub-pool runs.
 */
async function loadExamples(opts?: LoadExamplesOptions): Promise<TaskExample[]> {
  const seed = opts?.seed ?? DEFAULT_SEED;
  const limit = opts?.limit;
  const POOL_CAP = 1000;

  let pool = readCache();
  if (pool === null) {
    pool = await fetchFromHuggingFace(POOL_CAP);
    writeCache(pool);
  }

  const shuffled = seededShuffle(pool, seed);
  if (limit === undefined) return shuffled;
  return shuffled.slice(0, Math.max(0, limit));
}

/**
 * Default export — the canonical reasoning `TaskDefinition`.
 */
const reasoningTask: TaskDefinition<ReasoningReference, string> = {
  name: "reasoning",
  description:
    "Grade-school math word problems (GSM8K main test split). Scoring: exact match on the numeric answer after '####'.",
  promptTemplate,
  loadExamples,
  parseOutput,
  score,
};

export default reasoningTask;
