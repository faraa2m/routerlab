// eval/tasks/summarization.ts — summarization task definition for routerlab.
//
// Dataset: XSum (Narayan et al., 2018, "Don't Give Me the Details, Just
// the Summary!").
//   Source:  https://huggingface.co/datasets/EdinburghNLP/xsum
//   Split:   test (~11,334 examples).
//   License: CC-BY-SA-4.0 — redistribution-compatible with attribution.
//   Notes:   Each row is a BBC News article + a single-sentence professional
//            summary. XSum is the canonical single-sentence summarization
//            benchmark.
//
// Scoring: ROUGE-L F1 (longest-common-subsequence based). Implemented in
// pure TypeScript with no external NLP deps. We follow the standard
// recipe: lowercase, tokenize on whitespace + punctuation, compute LCS
// length, then F1 from precision = lcs/|pred|, recall = lcs/|gold|.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import {
  datasetCachePath,
  seededShuffle,
  type LoadExamplesOptions,
  type TaskDefinition,
  type TaskExample,
} from "./_types.ts";

const CACHE_FILENAME = "summarization.jsonl";
const DEFAULT_SEED = 42;

/**
 * XSum reference payload. Stored as a single-sentence string — XSum is
 * single-reference by construction.
 */
export interface SummarizationReference {
  goldSummary: string;
}

/**
 * HF datasets-server returns XSum rows in this shape. We narrow only the
 * fields we actually read.
 */
interface XsumRow {
  id: string;
  document: string;
  summary: string;
}

interface DatasetsServerResponse {
  rows: Array<{ row: XsumRow }>;
}

/**
 * Render the summarization prompt. Pure: depends only on `input`. We ask
 * for a single sentence to match the XSum reference distribution. Models
 * that produce paragraphs will be penalized by ROUGE-L on a single-
 * sentence gold (this is intentional — summarization quality is partly
 * the discipline of not over-producing).
 */
export function promptTemplate(input: TaskExample["input"]): string {
  const document = input.document ?? "";
  return `Summarize the following article in a single sentence.

Article: ${document}

Summary:`;
}

/**
 * Parse a raw model completion: trim, take the first non-empty line. The
 * "Summary:" label is sometimes echoed by smaller models; strip it if
 * present at the head of the line.
 */
export function parseOutput(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.length === 0) return "";
  const firstLine = trimmed.split(/\r?\n/).find((l) => l.trim().length > 0) ?? "";
  // Some models echo the label ("Summary: <actual summary>"). Strip a
  // leading "Summary:" (case-insensitive) so it doesn't pollute ROUGE.
  return firstLine.replace(/^\s*summary\s*:\s*/i, "").trim();
}

/**
 * Lowercase + simple whitespace/punctuation tokenizer. Mirrors the
 * normalization used by `rouge_score` (the de-facto reference impl) for
 * its `tokenize_text` step: ascii lowercase, strip punctuation, split on
 * whitespace. We use a Unicode-aware regex so non-ASCII letters survive.
 *
 * This is intentionally a *simple* tokenizer — ROUGE was designed to be
 * robust against tokenizer choice and our LCS-based scoring is dominated
 * by content-word overlap, not punctuation.
 */
export function tokenize(text: string): string[] {
  const lower = text.toLowerCase();
  // Replace anything that isn't a unicode letter/number/underscore with a
  // space, then split on whitespace.
  const stripped = lower.replace(/[^\p{L}\p{N}_]+/gu, " ");
  return stripped.split(/\s+/).filter((t) => t.length > 0);
}

/**
 * Length of the longest common subsequence of two token arrays. Standard
 * O(|a| * |b|) DP. Pure.
 *
 * Memory note: we use a two-row rolling buffer (O(min(|a|, |b|))) to keep
 * memory bounded for long summaries. Not strictly necessary for XSum
 * (single-sentence references rarely exceed 50 tokens), but the long-
 * document direction (a model that produces a paragraph) can blow up the
 * full-DP allocation.
 */
export function lcsLength(a: readonly string[], b: readonly string[]): number {
  if (a.length === 0 || b.length === 0) return 0;
  // Iterate over the shorter array as the inner loop dimension so the
  // rolling buffer stays small.
  const [outer, inner] = a.length >= b.length ? [a, b] : [b, a];
  const innerLen = inner.length;
  // Two rows of length innerLen + 1. Initialized to zero by default.
  let prev = new Array<number>(innerLen + 1).fill(0);
  let curr = new Array<number>(innerLen + 1).fill(0);
  for (let i = 1; i <= outer.length; i++) {
    for (let j = 1; j <= innerLen; j++) {
      // Indices i-1, j-1 are in bounds; noUncheckedIndexedAccess wants !.
      if (outer[i - 1]! === inner[j - 1]!) {
        curr[j] = (prev[j - 1] ?? 0) + 1;
      } else {
        const left = curr[j - 1] ?? 0;
        const up = prev[j] ?? 0;
        curr[j] = left > up ? left : up;
      }
    }
    const tmp = prev;
    prev = curr;
    curr = tmp;
    // Reset the (now stale) curr row to zero for the next iteration.
    curr.fill(0);
  }
  return prev[innerLen] ?? 0;
}

/**
 * ROUGE-L F1 between a prediction and a reference, both raw strings. Both
 * are tokenized with `tokenize()`, the LCS length is computed, and we
 * derive precision / recall / F1 from it. Returns a value in [0, 1].
 *
 * Edge cases:
 *   - Either side empty after tokenization → 0.0 (no overlap possible).
 *   - LCS == 0 → 0.0 (avoids division by zero in F1).
 *   - Both empty → 1.0 (trivially perfect, mirrors the SQuAD F1
 *     convention used by router-tasks-A).
 */
export function rougeLF1(prediction: string, reference: string): number {
  const predTokens = tokenize(prediction);
  const refTokens = tokenize(reference);
  if (predTokens.length === 0 && refTokens.length === 0) return 1;
  if (predTokens.length === 0 || refTokens.length === 0) return 0;
  const lcs = lcsLength(predTokens, refTokens);
  if (lcs === 0) return 0;
  const precision = lcs / predTokens.length;
  const recall = lcs / refTokens.length;
  return (2 * precision * recall) / (precision + recall);
}

/**
 * Score a parsed summary against an XSum reference. Single-reference
 * ROUGE-L F1, no max/min over a list (XSum is single-reference).
 */
export function score(
  parsed: string,
  reference: SummarizationReference,
): number {
  return rougeLF1(parsed, reference.goldSummary);
}

/**
 * Convert an XSum row from HF datasets-server into our `TaskExample` shape.
 */
function toTaskExample(row: XsumRow): TaskExample {
  const reference: SummarizationReference = { goldSummary: row.summary };
  return {
    id: row.id,
    input: { document: row.document },
    reference,
  };
}

/**
 * Read cached jsonl examples. Returns `null` on cache miss.
 */
function readCache(): TaskExample[] | null {
  const path = datasetCachePath(CACHE_FILENAME);
  if (!existsSync(path)) return null;
  const text = readFileSync(path, "utf8");
  const lines = text.split("\n").filter((l) => l.length > 0);
  return lines.map((line) => JSON.parse(line) as TaskExample);
}

/**
 * Persist examples to the on-disk cache.
 */
function writeCache(examples: TaskExample[]): void {
  const path = datasetCachePath(CACHE_FILENAME);
  mkdirSync(dirname(path), { recursive: true });
  const lines = examples.map((ex) => JSON.stringify(ex));
  writeFileSync(path, `${lines.join("\n")}\n`, "utf8");
}

/**
 * Fetch a batch of XSum test rows from HF datasets-server. Pool capped at
 * 1,000 — enough for routerlab's per-task-class frontier and small
 * enough to fit in a single overnight run within free-tier limits.
 */
async function fetchFromHuggingFace(cap: number): Promise<TaskExample[]> {
  const url = (offset: number, length: number): string =>
    `https://datasets-server.huggingface.co/rows?dataset=EdinburghNLP%2Fxsum&config=default&split=test&offset=${offset}&length=${length}`;
  const PAGE = 100;
  const collected: TaskExample[] = [];
  let offset = 0;
  while (collected.length < cap) {
    const remaining = cap - collected.length;
    const length = Math.min(PAGE, remaining);
    const resp = await fetch(url(offset, length));
    if (!resp.ok) {
      throw new Error(
        `HF datasets-server returned ${resp.status} for XSum (offset=${offset}, length=${length})`,
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
 * Resolve the examples backing this task. Cache first, network on miss,
 * then deterministic shuffle + limit. Pool capped at 1,000.
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
 * Default export — the canonical summarization `TaskDefinition`.
 */
const summarizationTask: TaskDefinition<SummarizationReference, string> = {
  name: "summarization",
  description:
    "Single-sentence article summarization (XSum test split). Scoring: ROUGE-L F1 against the BBC reference summary, pure-TS LCS implementation.",
  promptTemplate,
  loadExamples,
  parseOutput,
  score,
};

export default summarizationTask;
