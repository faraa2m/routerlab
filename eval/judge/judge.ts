// eval/judge/judge.ts — LLM-as-judge harness with prompt-hash caching.
//
// Used by `router-frontier` (Phase 3 sibling) to score model outputs at
// scale on task classes where reference metrics under-measure quality
// (chiefly codegen + summarization). The default judge is Anthropic's
// `claude-haiku-4-5` — the cheapest competent option in the catalog per
// the project plan.
//
// Design notes:
//   - The cache key is a SHA-256 of the canonical JSON encoding of
//     (judgeModel, taskClass, prompt, candidate, reference, rubric). Any
//     change to any of those fields produces a different key, so the cache
//     is correctly invalidated when a rubric is updated.
//   - Cache files live at `eval/judge/cache/{first2}/{rest}.json` (two-
//     char fanout to avoid massive single-dir scans on big sweeps).
//   - Cache-hit returns immediately — no API call.
//   - The underlying Anthropic SDK is used directly (not via the runner)
//     because the judge needs slightly different defaults (lower
//     `max_tokens`, lower temperature) and doesn't need the cost / token
//     accounting the runner adds. Retry policy is the same shared
//     `withRetries` primitive every runner uses.
//   - Score parsing is two-pass: first regex `Score:\s*(\d+(?:\.\d+)?)`,
//     fallback "last number in the response". Both paths divide by 10 and
//     clamp to [0,1].

import Anthropic from "@anthropic-ai/sdk";
import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { withRetries } from "../runners/_retry.ts";
import {
  makeRunnerError,
  RUNNER_DEFAULTS,
  type RunnerError,
} from "../runners/_types.ts";
import {
  DEFAULT_CACHE_DIR_RELATIVE,
  DEFAULT_JUDGE_MODEL,
  type JudgeOptions,
  type JudgeRequest,
  type JudgeResponse,
} from "./_types.ts";
import { buildJudgePrompt } from "./prompts.ts";

/**
 * Minimal shape we need from the Anthropic SDK response, mirroring the
 * shape used in `eval/runners/anthropic.ts`. Kept local so we don't depend
 * on internal SDK types.
 */
interface AnthropicMessageResponse {
  content: Array<{ type: string; text?: string }>;
  usage?: { input_tokens?: number; output_tokens?: number };
}

/**
 * SDK-like error shape we narrow into when classifying for retry. Same
 * idea as the anthropic runner's classifier.
 */
interface AnthropicLikeError {
  status?: number;
  message?: string;
  name?: string;
}

function isAnthropicLikeError(e: unknown): e is AnthropicLikeError {
  return typeof e === "object" && e !== null && ("status" in e || "message" in e);
}

/**
 * Classify an error from the judge's underlying anthropic call so the
 * retry policy can decide whether to retry. Same mapping as the anthropic
 * runner: 401/403 = auth (not retryable), 429 = rate_limit (retryable),
 * 5xx = server (retryable), AbortError = timeout (retryable), else
 * bad_request / unknown.
 */
function classifyJudgeError(e: unknown): RunnerError {
  const isAbort =
    e instanceof Error && (e.name === "AbortError" || e.message.includes("aborted"));
  if (isAbort) {
    return makeRunnerError("anthropic-judge", "timeout", "Judge request timed out", true);
  }
  if (isAnthropicLikeError(e)) {
    const status = e.status ?? 0;
    const msg = e.message ?? "Judge request failed";
    if (status === 401 || status === 403) {
      return makeRunnerError("anthropic-judge", "auth", msg, false);
    }
    if (status === 429) {
      return makeRunnerError("anthropic-judge", "rate_limit", msg, true);
    }
    if (status >= 500) {
      return makeRunnerError("anthropic-judge", "server", msg, true);
    }
    if (status >= 400) {
      return makeRunnerError("anthropic-judge", "bad_request", msg, false);
    }
  }
  const fallbackMsg =
    e instanceof Error ? e.message : "Unknown error from judge harness";
  return makeRunnerError("anthropic-judge", "unknown", fallbackMsg, false);
}

/**
 * Build the canonical key payload for a request. Field order is fixed so
 * the resulting JSON (and therefore the SHA) is stable across runs and
 * machines.
 *
 * `reference` is `unknown` from the caller; we JSON-encode it (with a
 * deterministic stringifier for plain objects via `sortedStringify` so
 * `{a:1,b:2}` and `{b:2,a:1}` hash to the same key).
 */
interface CacheKeyPayload {
  judgeModel: string;
  taskClass: string;
  prompt: string;
  candidate: string;
  reference: string;
  rubric: string;
}

/**
 * Deterministic JSON stringifier: objects emit keys in sorted order so
 * structurally-equal inputs produce byte-equal output. Arrays and
 * primitives use native semantics. We avoid pulling in `safe-stable-
 * stringify` to keep dependency surface zero.
 *
 * Cycle safety: we don't expect cycles in eval references but track a
 * visited set just in case — recursion on a cycle would otherwise blow
 * the stack.
 */
export function sortedStringify(value: unknown): string {
  const seen = new WeakSet<object>();
  const walk = (v: unknown): unknown => {
    if (v === null || typeof v !== "object") return v;
    if (seen.has(v as object)) return "[cycle]";
    seen.add(v as object);
    if (Array.isArray(v)) {
      return v.map(walk);
    }
    const obj = v as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(obj).sort()) {
      out[k] = walk(obj[k]);
    }
    return out;
  };
  return JSON.stringify(walk(value));
}

/**
 * Compute the cache key for a request. Exported so tests can assert
 * stability of the key composition.
 */
export function cacheKeyForRequest(
  req: JudgeRequest,
  judgeModel: string,
): string {
  const payload: CacheKeyPayload = {
    judgeModel,
    taskClass: req.taskClass,
    prompt: req.prompt,
    candidate: req.candidate,
    reference: sortedStringify(req.reference),
    rubric: req.rubric ?? "",
  };
  return createHash("sha256").update(sortedStringify(payload)).digest("hex");
}

/**
 * Resolve the on-disk cache path for a key. Two-char fanout: a 1k-call
 * sweep won't pile 1000 files into a single dir, and the fanout dirs are
 * easy to inspect manually.
 */
export function cacheFilePathForKey(cacheDir: string, key: string): string {
  const fanout = key.slice(0, 2);
  const rest = key.slice(2);
  return join(cacheDir, fanout, `${rest}.json`);
}

/**
 * Resolve the default cache directory relative to the eval/judge module.
 * We pin to `<repo>/eval/judge/cache` so the path is stable regardless of
 * the caller's CWD.
 */
function defaultCacheDir(): string {
  const here = dirname(fileURLToPath(import.meta.url));
  // eval/judge/judge.ts -> eval/judge/ -> eval/ -> repo root
  const repoRoot = resolve(here, "..", "..");
  return join(repoRoot, DEFAULT_CACHE_DIR_RELATIVE);
}

/**
 * Read a cached `JudgeResponse` for a key. Returns `null` if the cache
 * file is missing or malformed (corrupt-cache files are silently bypassed
 * and overwritten on the next live call so a bad file doesn't poison the
 * whole run).
 */
export function readCachedResponse(
  cacheDir: string,
  key: string,
): JudgeResponse | null {
  const path = cacheFilePathForKey(cacheDir, key);
  if (!existsSync(path)) return null;
  try {
    const raw = readFileSync(path, "utf8");
    const parsed = JSON.parse(raw) as JudgeResponse;
    // Defensive shape check — a stray file shouldn't crash the harness.
    if (
      typeof parsed.score !== "number" ||
      typeof parsed.reasoning !== "string" ||
      typeof parsed.judge_model !== "string" ||
      typeof parsed.ts !== "string"
    ) {
      return null;
    }
    return { ...parsed, cacheHit: true };
  } catch {
    return null;
  }
}

/**
 * Persist a `JudgeResponse` to the on-disk cache. Creates parent dirs on
 * demand. We deliberately store `cacheHit: false` in the file so a future
 * re-read can rewrite it to `true` (the cache-hit flag is a property of
 * the access, not the response).
 */
export function writeCachedResponse(
  cacheDir: string,
  key: string,
  resp: JudgeResponse,
): void {
  const path = cacheFilePathForKey(cacheDir, key);
  mkdirSync(dirname(path), { recursive: true });
  const onDisk: JudgeResponse = { ...resp, cacheHit: false };
  writeFileSync(path, JSON.stringify(onDisk, null, 2), "utf8");
}

/**
 * Concatenate the text blocks of an Anthropic SDK response. Same helper as
 * in the anthropic runner — inlined to avoid an import cycle.
 */
function extractText(resp: AnthropicMessageResponse): string {
  const parts: string[] = [];
  for (const block of resp.content) {
    if (block.type === "text" && typeof block.text === "string") {
      parts.push(block.text);
    }
  }
  return parts.join("");
}

/**
 * Parse a numeric score (0-10) out of the judge model's raw response.
 *
 * Strategy:
 *   1. Primary: match `Score:\s*(\d+(?:\.\d+)?)` (case-insensitive, multi-
 *      line). The judge prompt instructs the model to emit exactly this
 *      form on the final line.
 *   2. Fallback: extract the **last** numeric run in the response and
 *      interpret it as the 0-10 score. This catches malformed outputs
 *      where the model omitted the `Score:` label but still produced a
 *      number at the end.
 *   3. Last resort: 0.
 *
 * Returns the parsed number clamped to [0,10]. The caller divides by 10
 * to get a [0,1] score.
 */
export function parseRawScore(raw: string): number {
  const labelled = raw.match(/Score\s*:\s*(\d+(?:\.\d+)?)/i);
  if (labelled !== null && labelled[1] !== undefined) {
    const n = Number(labelled[1]);
    if (Number.isFinite(n)) return clamp(n, 0, 10);
  }
  // Fallback: last numeric run in the string.
  const numbers = raw.match(/\d+(?:\.\d+)?/g);
  if (numbers !== null && numbers.length > 0) {
    const last = numbers[numbers.length - 1];
    if (last !== undefined) {
      const n = Number(last);
      if (Number.isFinite(n)) return clamp(n, 0, 10);
    }
  }
  return 0;
}

/**
 * Extract the judge's reasoning text — everything except the trailing
 * `Score: …` line, trimmed. If no `Score:` line is present, returns the
 * entire response trimmed (best-effort).
 */
export function extractReasoning(raw: string): string {
  const match = raw.match(/^(.*?)(?:\n\s*)?Score\s*:\s*\d+(?:\.\d+)?\s*$/is);
  if (match !== null && match[1] !== undefined) {
    return match[1].trim();
  }
  return raw.trim();
}

/**
 * Clamp `x` into the inclusive range `[lo, hi]`. Exported for tests.
 */
export function clamp(x: number, lo: number, hi: number): number {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

/**
 * Main entrypoint. Returns a `JudgeResponse` either from cache or from a
 * fresh judge call.
 *
 * Behavior contract:
 *   - Cache hit: no API call; `cacheHit: true` in the response.
 *   - Cache miss: build prompt -> call judge -> parse score -> write
 *     cache -> return with `cacheHit: false`.
 *   - `opts.noCache: true` bypasses both read and write — useful when
 *     iterating on judge prompts and wanting to invalidate everything.
 *
 * `apiKey` is read from `ANTHROPIC_API_KEY` when not supplied. If neither
 * `apiKey` nor `client` is supplied and the env var is missing, the
 * function throws synchronously — fail-fast.
 */
export async function judge(
  request: JudgeRequest,
  opts: JudgeOptions = {},
): Promise<JudgeResponse> {
  const judgeModel = opts.judgeModel ?? DEFAULT_JUDGE_MODEL;
  const cacheDir = opts.cacheDir ?? defaultCacheDir();
  const key = cacheKeyForRequest(request, judgeModel);

  // Cache read path.
  if (opts.noCache !== true) {
    const cached = readCachedResponse(cacheDir, key);
    if (cached !== null) {
      return cached;
    }
  }

  // Live call path. Build the prompt and call the judge model.
  const prompt = buildJudgePrompt(request);
  const apiKey = opts.apiKey ?? process.env["ANTHROPIC_API_KEY"];
  if (opts.client === undefined && (apiKey === undefined || apiKey === "")) {
    throw new Error(
      "Judge requires ANTHROPIC_API_KEY (set env var or pass `apiKey`/`client`).",
    );
  }

  const client =
    opts.client ??
    (new Anthropic({ apiKey: apiKey as string }) as unknown as {
      messages: { create: (...args: unknown[]) => Promise<unknown> };
    });

  const timeoutMs = opts.timeoutMs ?? RUNNER_DEFAULTS.timeoutMs;
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);

  try {
    const resp = (await withRetries(
      async () => {
        try {
          return (await client.messages.create(
            {
              model: judgeModel,
              // Judge replies are short — reasoning + a one-line score.
              // 256 tokens is plenty; keeps cost bounded.
              max_tokens: 256,
              // Determinism is more important than diversity for grading.
              temperature: 0,
              messages: [{ role: "user", content: prompt }],
            },
            { signal: ac.signal },
          )) as unknown as AnthropicMessageResponse;
        } catch (e) {
          throw classifyJudgeError(e);
        }
      },
      {
        maxAttempts: opts.retry?.maxAttempts ?? RUNNER_DEFAULTS.retry.maxAttempts,
        baseDelayMs: opts.retry?.baseDelayMs ?? RUNNER_DEFAULTS.retry.baseDelayMs,
        jitterPct: opts.retry?.jitterPct ?? RUNNER_DEFAULTS.retry.jitterPct,
        sleepFn: opts.retry?.sleepFn,
      },
    )) as AnthropicMessageResponse;

    const rawText = extractText(resp);
    const rawScore = parseRawScore(rawText);
    const score = rawScore / 10;
    const reasoning = extractReasoning(rawText);

    const response: JudgeResponse = {
      score,
      reasoning,
      judge_model: judgeModel,
      cacheHit: false,
      ts: new Date().toISOString(),
    };

    if (opts.noCache !== true) {
      writeCachedResponse(cacheDir, key, response);
    }
    return response;
  } finally {
    clearTimeout(timer);
  }
}
