// eval/tasks/classification.ts — 3-class sentiment classification task.
//
// Dataset: TweetEval (Barbieri et al., 2020, EMNLP Findings),
//          sentiment subset, test split.
//   Source:  https://huggingface.co/datasets/cardiffnlp/tweet_eval
//   License: Apache-2.0 — fully redistribution-compatible.
//   Labels:  0 = negative, 1 = neutral, 2 = positive.
//   Size:    test split has 12,284 tweets.
//
// Scoring: exact match against the gold label after the lightweight
// alias normalization in `parseOutput`. Exact match is the canonical
// metric for sentiment classification in TweetEval's own paper.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import {
  datasetCachePath,
  seededShuffle,
  type LoadExamplesOptions,
  type TaskDefinition,
  type TaskExample,
} from "./_types.ts";

const CACHE_FILENAME = "classification.jsonl";
const DEFAULT_SEED = 42;

/**
 * The three canonical sentiment labels. Keep this list closed — any
 * widening would silently change `score` semantics for downstream callers.
 */
export type SentimentLabel = "negative" | "neutral" | "positive";

const LABELS: readonly SentimentLabel[] = ["negative", "neutral", "positive"];

/**
 * Aliases the parser accepts and normalizes to a canonical label. Kept
 * narrow on purpose — accepting "good" / "bad" would conflate sentiment
 * with quality judgement, which is exactly what we don't want.
 */
const ALIASES: ReadonlyMap<string, SentimentLabel> = new Map<
  string,
  SentimentLabel
>([
  ["negative", "negative"],
  ["neg", "negative"],
  ["-", "negative"],
  ["neutral", "neutral"],
  ["neu", "neutral"],
  ["mixed", "neutral"],
  ["none", "neutral"],
  ["positive", "positive"],
  ["pos", "positive"],
  ["+", "positive"],
]);

/**
 * Raw HF datasets-server row shape for TweetEval's sentiment config. We
 * narrow to the fields we use.
 */
interface TweetEvalRow {
  text: string;
  label: number;
}

interface DatasetsServerResponse {
  rows: Array<{ row: TweetEvalRow; row_idx: number }>;
}

/**
 * Render the classification prompt. Pure: input -> string. We anchor on
 * "exactly one of" to push models toward terse, parseable outputs.
 */
export function promptTemplate(input: TaskExample["input"]): string {
  const text = input.text ?? "";
  return `Classify this tweet's sentiment as exactly one of: negative, neutral, positive.

Tweet: ${text}

Sentiment:`;
}

/**
 * Parse a raw completion into a `SentimentLabel`, or `null` if the model
 * produced something we can't safely map. Returning `null` is preferable
 * to guessing — scoring treats unparseable outputs as wrong, which is the
 * honest reading.
 *
 * Strategy: lowercase + strip non-letters, take the first non-empty
 * token, look it up in the alias table.
 */
export function parseOutput(raw: string): SentimentLabel | null {
  const lower = raw.toLowerCase().trim();
  if (lower.length === 0) return null;
  const firstLine = lower.split(/\r?\n/)[0] ?? "";
  // Split on anything that isn't a letter or the +/- symbols we accept.
  const tokens = firstLine.split(/[^a-z+\-]+/).filter((t) => t.length > 0);
  for (const token of tokens) {
    const hit = ALIASES.get(token);
    if (hit !== undefined) return hit;
  }
  return null;
}

/**
 * Exact-match score in {0, 1}. Unparseable predictions (parsed === null)
 * are wrong by definition.
 */
export function score(
  parsed: SentimentLabel | null,
  reference: SentimentLabel,
): number {
  if (parsed === null) return 0;
  return parsed === reference ? 1 : 0;
}

/**
 * Map a TweetEval numeric label to the canonical string label. The
 * dataset's label2id schema is fixed across versions, so this map is safe.
 */
function labelFromIndex(idx: number): SentimentLabel {
  const label = LABELS[idx];
  if (label === undefined) {
    throw new Error(`TweetEval sentiment label index out of range: ${idx}`);
  }
  return label;
}

/**
 * Translate one HF row into our `TaskExample` shape.
 */
function toTaskExample(row: TweetEvalRow, idx: number): TaskExample {
  const label = labelFromIndex(row.label);
  return {
    id: `tweet_eval_sentiment_test_${idx}`,
    input: { text: row.text },
    reference: label,
    metadata: { labelIndex: row.label },
  };
}

/**
 * Read the on-disk cache (jsonl). Returns null if missing.
 */
function readCache(): TaskExample[] | null {
  const path = datasetCachePath(CACHE_FILENAME);
  if (!existsSync(path)) return null;
  const text = readFileSync(path, "utf8");
  const lines = text.split("\n").filter((l) => l.length > 0);
  return lines.map((line) => JSON.parse(line) as TaskExample);
}

/**
 * Persist examples to the cache.
 */
function writeCache(examples: TaskExample[]): void {
  const path = datasetCachePath(CACHE_FILENAME);
  mkdirSync(dirname(path), { recursive: true });
  const lines = examples.map((ex) => JSON.stringify(ex));
  writeFileSync(path, `${lines.join("\n")}\n`, "utf8");
}

/**
 * Fetch TweetEval/sentiment test rows from HF datasets-server. Mirrors
 * the SQuAD fetcher's structure (page through 100/req up to `cap`).
 */
async function fetchFromHuggingFace(cap: number): Promise<TaskExample[]> {
  const url = (offset: number, length: number): string =>
    `https://datasets-server.huggingface.co/rows?dataset=cardiffnlp%2Ftweet_eval&config=sentiment&split=test&offset=${offset}&length=${length}`;
  const pageSize = 100;
  const collected: TaskExample[] = [];
  let offset = 0;
  while (collected.length < cap) {
    const remaining = cap - collected.length;
    const length = Math.min(pageSize, remaining);
    const resp = await fetch(url(offset, length));
    if (!resp.ok) {
      throw new Error(
        `HF datasets-server returned ${resp.status} for TweetEval (offset=${offset}, length=${length})`,
      );
    }
    const json = (await resp.json()) as DatasetsServerResponse;
    if (json.rows.length === 0) break;
    for (const entry of json.rows) {
      collected.push(toTaskExample(entry.row, entry.row_idx));
    }
    offset += json.rows.length;
    if (json.rows.length < length) break;
  }
  return collected;
}

/**
 * Load (and cache) a deterministically-shuffled subset of TweetEval/test.
 * Pool cap of 1,000 is plenty for per-task-class frontier work.
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
 * Default export — the canonical classification `TaskDefinition`. Wire
 * this into runners and frontier construction the same way as `qa`.
 */
const classificationTask: TaskDefinition<SentimentLabel, SentimentLabel | null> = {
  name: "classification",
  description:
    "3-class tweet sentiment (negative / neutral / positive) from TweetEval (Apache-2.0). Scoring: exact match against the gold label.",
  promptTemplate,
  loadExamples,
  parseOutput,
  score,
};

export default classificationTask;
