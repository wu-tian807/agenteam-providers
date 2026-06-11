// @desc Shared shape helper: inline text_file parts as utf-8 text with head/tail truncation.

import type { ContentPart, LLMMessage } from "@agenteam/types";
import type { MediaReaders } from "./media-readers.js";

export interface InlineTextFilesOptions {
  /** Files this size or smaller are inlined whole. Default: 256_000 bytes
   *  (≈ what the `read_file` tool surfaces by default — keeps shape output
   *  predictable for users used to the tool). */
  maxInlineBytes?: number;
  /** Head bytes preserved when a file exceeds `maxInlineBytes`. Default: 160_000. */
  headBytes?: number;
  /** Tail bytes preserved. Default: 64_000. */
  tailBytes?: number;
}

const DEFAULT_MAX_INLINE_BYTES = 256_000;
const DEFAULT_HEAD_BYTES = 160_000;
const DEFAULT_TAIL_BYTES = 64_000;

/**
 * Replace every `text_file` ContentPart with a `text` part carrying the
 * file's utf-8 body. This is the shared shape step for **all** providers —
 * the `text_file` part type never reaches a `convertPartTo*` function.
 *
 * Truncation contract:
 *   - `bytes <= maxInlineBytes` → entire body inlined.
 *   - `bytes >  maxInlineBytes` → head + truncation marker + tail. Both
 *     ends are aligned to utf-8 character boundaries (no torn multi-byte
 *     sequences) using a TextDecoder with `fatal: false`.
 *
 * Read failure is non-fatal — emits `[file unavailable: <path>]` so the
 * surrounding tool_result/user message keeps its shape and the model can
 * still reason about what was attempted.
 */
export async function inlineTextFiles(
  messages: LLMMessage[],
  readers: MediaReaders,
  opts: InlineTextFilesOptions = {},
): Promise<LLMMessage[]> {
  const maxInline = opts.maxInlineBytes ?? DEFAULT_MAX_INLINE_BYTES;
  const headBytes = opts.headBytes ?? DEFAULT_HEAD_BYTES;
  const tailBytes = opts.tailBytes ?? DEFAULT_TAIL_BYTES;

  let touched = false;
  const out = await Promise.all(messages.map(async (msg) => {
    if (!Array.isArray(msg.content)) return msg;
    let msgTouched = false;
    const parts: ContentPart[] = [];
    for (const part of msg.content) {
      if (part.type !== "text_file") {
        parts.push(part);
        continue;
      }
      msgTouched = true;
      try {
        const { bytes } = await readers.readFileBytes(part);
        const total = bytes.byteLength;
        if (total <= maxInline) {
          parts.push({ type: "text", text: `[file: ${part.path}]\n${bytes.toString("utf8")}` });
          continue;
        }
        const head = decodeBoundaryAligned(bytes.subarray(0, headBytes), "head");
        const tail = decodeBoundaryAligned(bytes.subarray(total - tailBytes), "tail");
        const omitted = total - headBytes - tailBytes;
        const text =
          `[file: ${part.path}]\n` +
          `${head}\n` +
          `... [truncated: ${omitted} bytes omitted, total ${total}] ...\n` +
          `${tail}`;
        parts.push({ type: "text", text });
      } catch {
        parts.push({ type: "text", text: `[file unavailable: ${part.path}]` });
      }
    }
    if (!msgTouched) return msg;
    touched = true;
    return { ...msg, content: parts };
  }));
  return touched ? out : messages;
}

/**
 * Decode a Buffer slice with utf-8 boundary alignment.
 *   - `head`: drop incomplete trailing bytes (decoder with fatal:false +
 *     ignoreBOM:false handles invalid bytes by replacement; we also
 *     trim a known-bad suffix length when the slice ends mid-codepoint).
 *   - `tail`: drop incomplete leading bytes by skipping forward to the
 *     next valid utf-8 start byte (`(b & 0xC0) !== 0x80`).
 */
function decodeBoundaryAligned(slice: Buffer, side: "head" | "tail"): string {
  if (side === "tail") {
    // Skip continuation bytes at the start (0b10xxxxxx).
    let start = 0;
    while (start < slice.byteLength && (slice[start] & 0xC0) === 0x80) start++;
    return slice.subarray(start).toString("utf8");
  }
  // Head: trim incomplete trailing multi-byte sequence.
  let end = slice.byteLength;
  // Walk back over continuation bytes to find the start of the last sequence.
  let cont = 0;
  while (end > 0 && (slice[end - 1] & 0xC0) === 0x80) {
    end--;
    cont++;
  }
  if (end === 0) return slice.toString("utf8");
  const lead = slice[end - 1];
  // Determine expected sequence length from the lead byte.
  let expected = 1;
  if ((lead & 0xE0) === 0xC0) expected = 2;
  else if ((lead & 0xF0) === 0xE0) expected = 3;
  else if ((lead & 0xF8) === 0xF0) expected = 4;
  const have = cont + 1;
  if (have < expected) {
    // Incomplete sequence at end — drop it.
    return slice.subarray(0, end - 1).toString("utf8");
  }
  return slice.subarray(0, slice.byteLength).toString("utf8");
}
