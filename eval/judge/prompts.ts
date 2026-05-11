// eval/judge/prompts.ts — per-task-class judge prompt templates.
//
// Each template asks the judge model (default: Haiku 4.5) to score a
// candidate output 0-10 along with brief reasoning. The harness in
// `judge.ts` parses the score out and divides by 10 to get a [0,1] number.
//
// Templates follow a fixed shape:
//   1. One-line role declaration ("You are an expert grader …").
//   2. The task-specific rubric (callable from `DEFAULT_RUBRICS`).
//   3. The original prompt + candidate + (optional) gold reference.
//   4. A strict output-format instruction so parsing is robust.
//
// Important: the **last** line of the judge's output must be
// `Score: <integer 0-10>`. The parser primarily keys on a regex match for
// `^Score:\s*(\d+(?:\.\d+)?)` and falls back to "last number in the
// response" if the strict form is absent. Both paths divide by 10.
//
// No `any`, no template-literal `eval`, fully strict TS.

import type { TaskClass } from "../../packages/core/src/types.ts";
import type { JudgeRequest } from "./_types.ts";

/**
 * Per-task-class default rubrics. Each rubric is a short paragraph
 * describing what "good" looks like for that task class. Callers can
 * override via `JudgeRequest.rubric`.
 *
 * These are intentionally tight — long rubrics make the judge slower (more
 * input tokens) without measurably improving agreement with humans, per
 * the LLM-as-judge literature (Zheng et al., 2023, "Judging LLM-as-a-Judge"
 * arXiv:2306.05685).
 */
export const DEFAULT_RUBRICS: Readonly<Record<TaskClass, string>> = {
  qa: [
    "Score how accurately the candidate answers the question grounded in the passage.",
    "10 = answer is correct and minimal; 7-9 = correct but slightly verbose or imprecise;",
    "4-6 = partially correct or partially supported; 1-3 = mostly wrong but on-topic;",
    "0 = wrong, hallucinated, or refused when an answer is supported by the passage.",
    "If the gold reference indicates the question is unanswerable from the passage,",
    "an explicit abstain (\"no answer\", \"unanswerable\") scores 10; a hallucinated answer scores 0.",
  ].join(" "),
  codegen: [
    "Score the candidate Python function body for correctness and quality.",
    "10 = code is correct, idiomatic, and would pass the implied test suite;",
    "7-9 = correct logic with minor style issues; 4-6 = approximately correct but with",
    "bugs that would fail edge cases; 1-3 = wrong logic but right shape;",
    "0 = doesn't compile, is malicious, or solves a different problem.",
    "Penalize markdown fences, explanations, or extra imports that the prompt did not request.",
  ].join(" "),
  summarization: [
    "Score how well the candidate summary captures the key points of the article in a single sentence.",
    "10 = faithful, concise, and informative; 7-9 = mostly faithful with small omissions or extra detail;",
    "4-6 = captures gist but with notable factual drift or missing key fact;",
    "1-3 = significant hallucination or wrong topic; 0 = unrelated or refused.",
    "Penalize multi-sentence outputs proportional to how much they exceed a single sentence.",
  ].join(" "),
  classification: [
    "Score whether the candidate output names the correct class label.",
    "10 = exactly the correct label; 7-9 = correct intent in slightly different wording;",
    "4-6 = related but distinct label; 1-3 = wrong but plausible label; 0 = wrong / off-topic / refused.",
  ].join(" "),
  reasoning: [
    "Score the candidate's reasoning trace and final answer on a multi-step problem.",
    "10 = correct final answer with sound reasoning; 7-9 = correct answer with minor reasoning gaps;",
    "4-6 = correct intermediate steps but wrong final answer (or vice versa);",
    "1-3 = wrong reasoning and wrong answer; 0 = unrelated, hallucinated, or refused.",
    "Weight the final answer more than the trace — a correct answer with terse reasoning still scores well.",
  ].join(" "),
};

/**
 * Stringify a gold reference for inclusion in the judge prompt. Each task
 * class encodes its reference differently, so we render a sensible textual
 * representation rather than dumping JSON for everything (which would hurt
 * judge agreement on tasks where the gold is itself plain text).
 *
 * The rule:
 *   - `undefined` -> empty marker text
 *   - `string`    -> verbatim
 *   - everything else -> JSON.stringify, fixed indent 2
 */
export function renderReference(reference: unknown): string {
  if (reference === undefined || reference === null) {
    return "(no gold reference provided)";
  }
  if (typeof reference === "string") {
    return reference;
  }
  try {
    return JSON.stringify(reference, null, 2);
  } catch {
    return String(reference);
  }
}

/**
 * Build the judge prompt for one request. Uses the request's `rubric`
 * override if present, otherwise the task class's default.
 *
 * The output instruction at the end is load-bearing for the parser in
 * `judge.ts` — change it only in tandem with the parser regex.
 */
export function buildJudgePrompt(req: JudgeRequest): string {
  const rubric = req.rubric ?? DEFAULT_RUBRICS[req.taskClass];
  const referenceText = renderReference(req.reference);

  return [
    `You are an expert grader for task class "${req.taskClass}".`,
    "",
    "RUBRIC:",
    rubric,
    "",
    "ORIGINAL PROMPT (given to candidate model):",
    req.prompt,
    "",
    "CANDIDATE OUTPUT:",
    req.candidate,
    "",
    "GOLD REFERENCE (may be empty for ungrounded tasks):",
    referenceText,
    "",
    "Reason briefly in 1-3 sentences, then on the final line output exactly:",
    "Score: <integer between 0 and 10>",
    "",
    "Your verdict:",
  ].join("\n");
}
