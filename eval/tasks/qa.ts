// eval/tasks/qa.ts — extractive QA task definition for routerlab.
//
// Dataset: SQuAD v2 (Rajpurkar et al., 2018, "Know What You Don't Know").
//   Source:  https://huggingface.co/datasets/rajpurkar/squad_v2
//   Split:   validation (11,873 examples).
//   License: CC-BY-SA-4.0 — redistribution-compatible with attribution.
//   Notes:   v2 augments v1 with unanswerable questions; gold answer is
//            an empty string for those rows and the model is expected to
//            say "no answer" (or similar).
//
// Scoring: token-level F1 vs. the gold answer list, taking max across the
// list (standard SQuAD metric). Implemented in pure TS — no NLP deps.
// For unanswerable questions, we treat any of a fixed set of phrases as
// the correct "abstain" response.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import {
  datasetCachePath,
  seededShuffle,
  type LoadExamplesOptions,
  type TaskDefinition,
  type TaskExample,
} from "./_types.ts";

const CACHE_FILENAME = "qa.jsonl";
const DEFAULT_SEED = 42;

/**
 * SQuAD answer payload: zero-or-more gold answers + an is_impossible flag.
 * v2's empty `answers.text` list is the canonical "unanswerable" marker.
 */
export interface QaReference {
  goldAnswers: string[];
  isImpossible: boolean;
}

/**
 * HF datasets-server returns rows in this shape. We narrow only the
 * fields we actually read; the API returns more.
 */
interface SquadRow {
  id: string;
  context: string;
  question: string;
  answers: { text: string[]; answer_start: number[] };
}

interface DatasetsServerResponse {
  rows: Array<{ row: SquadRow }>;
}

/**
 * Phrases we treat as a valid "abstain" answer for unanswerable questions.
 * Kept short on purpose — anything more expansive risks rewarding models
 * that hedge on answerable questions too.
 */
const ABSTAIN_PHRASES: readonly string[] = [
  "no answer",
  "unanswerable",
  "cannot be answered",
  "cannot answer",
  "not answerable",
  "the passage does not say",
  "the passage doesn't say",
  "not stated",
  "i don't know",
  "i do not know",
];

/**
 * Render the QA prompt. Pure: depends only on `input`. We deliberately
 * keep the format minimal so we don't bias toward verbose models.
 */
export function promptTemplate(input: TaskExample["input"]): string {
  const context = input.context ?? "";
  const question = input.question ?? "";
  return `Read this passage and answer the question.

Passage: ${context}

Question: ${question}

Answer:`;
}

/**
 * Parse a raw model completion. Trim whitespace, take the first non-empty
 * line. Anything past a newline is treated as commentary the model
 * shouldn't have produced.
 */
export function parseOutput(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.length === 0) return "";
  const firstLine = trimmed.split(/\r?\n/)[0];
  return (firstLine ?? "").trim();
}

/**
 * SQuAD-standard normalization: lowercase, strip punctuation, drop the
 * articles a/an/the, collapse whitespace. Used by both F1 scoring and
 * abstain detection. Pure.
 *
 * This mirrors the official SQuAD eval script's `normalize_answer`.
 */
export function normalizeAnswer(text: string): string {
  const lower = text.toLowerCase();
  const noPunct = lower.replace(/[^\p{L}\p{N}\s]/gu, " ");
  const noArticles = noPunct.replace(/\b(a|an|the)\b/g, " ");
  return noArticles.replace(/\s+/g, " ").trim();
}

/**
 * Tokenize a normalized string into whitespace-separated tokens. Empty
 * string yields an empty token list, which the F1 code handles explicitly.
 */
function tokenize(text: string): string[] {
  const normed = normalizeAnswer(text);
  if (normed.length === 0) return [];
  return normed.split(" ");
}

/**
 * Token-level F1 between two strings, after SQuAD normalization. Returns
 * a value in [0, 1]. Edge cases mirror the official script:
 *   - both empty after normalization → 1.0 (perfect agreement on empty)
 *   - exactly one empty             → 0.0
 *   - no overlap                    → 0.0
 */
export function tokenF1(prediction: string, gold: string): number {
  const predTokens = tokenize(prediction);
  const goldTokens = tokenize(gold);

  if (predTokens.length === 0 && goldTokens.length === 0) return 1;
  if (predTokens.length === 0 || goldTokens.length === 0) return 0;

  const goldCounts = new Map<string, number>();
  for (const t of goldTokens) goldCounts.set(t, (goldCounts.get(t) ?? 0) + 1);

  let common = 0;
  for (const t of predTokens) {
    const c = goldCounts.get(t) ?? 0;
    if (c > 0) {
      common += 1;
      goldCounts.set(t, c - 1);
    }
  }

  if (common === 0) return 0;
  const precision = common / predTokens.length;
  const recall = common / goldTokens.length;
  return (2 * precision * recall) / (precision + recall);
}

/**
 * Detect whether a prediction looks like an "abstain" response. Used for
 * v2 unanswerable questions where the gold is `[]` and we want to give
 * credit to a model that correctly says it doesn't know.
 */
export function isAbstain(prediction: string): boolean {
  const normed = normalizeAnswer(prediction);
  if (normed.length === 0) return true;
  for (const phrase of ABSTAIN_PHRASES) {
    if (normed.includes(normalizeAnswer(phrase))) return true;
  }
  return false;
}

/**
 * Score a parsed prediction against a SQuAD reference.
 *   - If `isImpossible`: 1.0 iff the prediction is an abstain phrase.
 *   - Else: max token F1 across the gold answer list.
 *
 * The max-across-golds rule matches the official SQuAD eval and is what
 * leaderboard numbers report. Returning 0 for an empty gold list on an
 * answerable question is defensive — shouldn't happen with valid data.
 */
export function score(parsed: string, reference: QaReference): number {
  if (reference.isImpossible) {
    return isAbstain(parsed) ? 1 : 0;
  }
  if (reference.goldAnswers.length === 0) return 0;
  let best = 0;
  for (const gold of reference.goldAnswers) {
    const f1 = tokenF1(parsed, gold);
    if (f1 > best) best = f1;
  }
  return best;
}

/**
 * Internal: convert a HF datasets-server SQuAD row into our `TaskExample`
 * shape. Empty `answers.text` is the v2 unanswerable marker.
 */
function toTaskExample(row: SquadRow): TaskExample {
  const isImpossible = row.answers.text.length === 0;
  const reference: QaReference = {
    goldAnswers: row.answers.text,
    isImpossible,
  };
  return {
    id: row.id,
    input: {
      context: row.context,
      question: row.question,
    },
    reference,
    metadata: { isImpossible },
  };
}

/**
 * Read cached jsonl examples (one JSON-encoded `TaskExample` per line).
 * Returns `null` if the cache doesn't exist, signalling the caller to
 * fetch fresh from HF.
 */
function readCache(): TaskExample[] | null {
  const path = datasetCachePath(CACHE_FILENAME);
  if (!existsSync(path)) return null;
  const text = readFileSync(path, "utf8");
  const lines = text.split("\n").filter((l) => l.length > 0);
  return lines.map((line) => JSON.parse(line) as TaskExample);
}

/**
 * Persist examples to the on-disk cache, creating the directory on demand.
 * Format: jsonl (one example per line) — easy to grep, easy to stream.
 */
function writeCache(examples: TaskExample[]): void {
  const path = datasetCachePath(CACHE_FILENAME);
  mkdirSync(dirname(path), { recursive: true });
  const lines = examples.map((ex) => JSON.stringify(ex));
  writeFileSync(path, `${lines.join("\n")}\n`, "utf8");
}

/**
 * Fetch a batch of SQuAD v2 validation rows from HuggingFace's public
 * datasets-server. We page through up to `cap` rows (paginated 100/req)
 * and cache the union locally. Network errors propagate — caller decides
 * what to do.
 *
 * The datasets-server is the same backend the HF dataset preview UI uses;
 * it requires no auth for public datasets and serves Parquet-backed rows
 * as JSON. Reference: https://huggingface.co/docs/datasets-server
 */
async function fetchFromHuggingFace(cap: number): Promise<TaskExample[]> {
  const url = (offset: number, length: number): string =>
    `https://datasets-server.huggingface.co/rows?dataset=rajpurkar%2Fsquad_v2&config=squad_v2&split=validation&offset=${offset}&length=${length}`;
  const pageSize = 100;
  const collected: TaskExample[] = [];
  let offset = 0;
  while (collected.length < cap) {
    const remaining = cap - collected.length;
    const length = Math.min(pageSize, remaining);
    const resp = await fetch(url(offset, length));
    if (!resp.ok) {
      throw new Error(
        `HF datasets-server returned ${resp.status} for SQuAD v2 (offset=${offset}, length=${length})`,
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
 * Resolve the examples backing this task, fetching + caching on first
 * call. `limit` slices the deterministically-shuffled pool; `seed` controls
 * the shuffle. The pool size is capped at 1,000 — enough for routerlab's
 * per-task-class frontier and small enough to fetch in a single pass.
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
 * Default export — the canonical QA `TaskDefinition` consumed by runners,
 * the router engine, and the smoke script.
 */
const qaTask: TaskDefinition<QaReference, string> = {
  name: "qa",
  description:
    "Extractive QA on Wikipedia passages (SQuAD v2 validation split). Scoring: max token F1 across gold answers; abstain credit on unanswerable rows.",
  promptTemplate,
  loadExamples,
  parseOutput,
  score,
};

export default qaTask;
