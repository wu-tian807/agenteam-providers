// @desc Layer 1 framework-shared inbound text_file truncation (oversize parts -> head+tail text).
//
// Layer 1 of the two-layer `prepareInboundMessages` pipeline composed by
// `buildModelProvider`'s wrapper:
//
//   Layer 1 (this file): framework-shared lossy decisions, applied to every
//     provider. Currently: truncate any `text_file` part whose body exceeds
//     `MAX_INLINE_BYTES` into a head/tail slice carried inside a regular
//     `text` part (so it survives WAL serialisation as plain text and never
//     re-reads the file at wire time).
//
//   Layer 2 (provider's own `prepareInboundMessages`): provider-specific
//     side effects on top of Layer 1's view (e.g. Gemini's Files API upload
//     writing fileRefs into the message's sidecar). Subclasses receive the
//     already-shaped input and don't need to call `super`.
//
// Design choices:
//   * Small `text_file` parts (≤ 256 KB) are kept untouched. Each provider's
//     wire converter (`anthropic.ts`, `openai-compat.ts`, …) still inlines
//     them as plain UTF-8 — that small-file fallback is intentionally
//     preserved.
//   * Truncation aligns to UTF-8 codepoint boundaries on both ends so the
//     emitted text is always valid UTF-8 (no dangling continuation bytes).
//   * Read failures degrade to a `[file unavailable: …]` text part rather
//     than throwing, mirroring the wire-converter fallbacks.
//   * Pure function: no closure state, no caches. Re-running on already
//     shaped messages is idempotent (truncated text has no `text_file`
//     part left to match).
//
// Future framework-shared inbound decisions (e.g. image size cap) belong
// in this same file as additional helpers; the wrapper composes them by
// chaining function calls before the provider's own hook runs.

import type { ContentPart, LLMMessage } from "@agenteam/types";
import type { MediaReaders } from "./media-readers.js";

/** Bytes above this threshold trigger head/tail truncation. Aligned with the
 *  default `read_file` tool budget so the LLM-visible "max single-file size"
 *  is uniform across attachment and tool-read paths. */
export const MAX_INLINE_BYTES = 256_000;
export const HEAD_BYTES = 160_000;
export const TAIL_BYTES = 64_000;

export interface InboundTruncationOptions {
  maxInlineBytes?: number;
  headBytes?: number;
  tailBytes?: number;
}

/**
 * Apply framework-shared inbound truncation to a message list.
 *
 * Currently the only transformation is large-`text_file` truncation; this
 * function is intentionally narrow so the semantic boundary stays clear.
 *
 * @param messages  Inbound messages from the engine (post-WAL replay).
 * @param readers   Host-supplied media readers — same instance the wire
 *                  converters use, so path resolution stays consistent.
 * @param opts      Optional threshold overrides (mostly for tests).
 */
export async function applyInboundTruncation(
  messages: LLMMessage[],
  readers: MediaReaders,
  opts?: InboundTruncationOptions,
): Promise<LLMMessage[]> {
  const max = opts?.maxInlineBytes ?? MAX_INLINE_BYTES;
  const headBudget = opts?.headBytes ?? HEAD_BYTES;
  const tailBudget = opts?.tailBytes ?? TAIL_BYTES;

  let mutated = false;
  const out: LLMMessage[] = new Array(messages.length);
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    const content = msg.content;
    let newContent: ContentPart[] | null = null;
    for (let j = 0; j < content.length; j++) {
      const part = content[j];
      if (part.type !== "text_file") continue;
      const replacement = await truncateTextFilePart(part, readers, max, headBudget, tailBudget);
      if (!replacement) continue;
      if (!newContent) newContent = content.slice();
      newContent[j] = replacement;
    }
    if (newContent) {
      mutated = true;
      out[i] = { ...msg, content: newContent };
    } else {
      out[i] = msg;
    }
  }
  return mutated ? out : messages;
}

/** Returns a replacement part if the text_file should be truncated, or `null`
 *  if it stays unchanged. */
async function truncateTextFilePart(
  part: Extract<ContentPart, { type: "text_file" }>,
  readers: MediaReaders,
  max: number,
  headBudget: number,
  tailBudget: number,
): Promise<ContentPart | null> {
  let bytes: Buffer;
  try {
    bytes = (await readers.readFileBytes(part)).bytes;
  } catch {
    return { type: "text", text: `[file unavailable: ${part.path}]` };
  }
  if (bytes.length <= max) return null;

  const headEnd = alignHeadEnd(bytes, headBudget);
  const tailStart = alignTailStart(bytes, bytes.length - tailBudget);
  const head = bytes.subarray(0, headEnd).toString("utf8");
  const tail = bytes.subarray(tailStart).toString("utf8");
  const omitted = tailStart - headEnd;
  const text =
    `[file: ${part.path}]\n` +
    `${head}\n` +
    `... [truncated: ${omitted} bytes omitted, total ${bytes.length}] ...\n` +
    tail;
  return { type: "text", text };
}

/** Walk `headEnd` back to the nearest byte that does NOT continue a multi-byte
 *  UTF-8 sequence, so `bytes.subarray(0, headEnd)` decodes cleanly. */
function alignHeadEnd(bytes: Buffer, headEnd: number): number {
  if (headEnd >= bytes.length) return bytes.length;
  let end = headEnd;
  while (end > 0 && (bytes[end] & 0xc0) === 0x80) end--;
  return end;
}

/** Walk `tailStart` forward past any trailing continuation bytes so
 *  `bytes.subarray(tailStart)` starts on a codepoint boundary. */
function alignTailStart(bytes: Buffer, tailStart: number): number {
  let start = Math.max(0, tailStart);
  while (start < bytes.length && (bytes[start] & 0xc0) === 0x80) start++;
  return start;
}
