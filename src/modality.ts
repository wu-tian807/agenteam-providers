import type { ContentPart, InputModality } from "@agenteam/types";
import { isFileMediaContentPart } from "@agenteam/types";
import type { LLMMessage } from "./types.js";

// ─── Surrogate-pair sanitiser ────────────────────────────────────────────────
// Strips lone UTF-16 surrogates that would survive `JSON.stringify` and crash
// providers that re-encode the payload as UTF-8 (Anthropic / OpenAI bodies).
// Kept here, not behind a separate module, because every entry point that
// shapes content for a provider already imports something else from this file.

const REPLACEMENT_CHAR = "\uFFFD";

function isHighSurrogate(code: number): boolean {
  return code >= 0xd800 && code <= 0xdbff;
}

function isLowSurrogate(code: number): boolean {
  return code >= 0xdc00 && code <= 0xdfff;
}

export function sanitizeInvalidSurrogates(input: string): string {
  if (!input) return input;
  let out = "";
  for (let i = 0; i < input.length; i++) {
    const code = input.charCodeAt(i);
    if (isHighSurrogate(code)) {
      if (i + 1 < input.length) {
        const next = input.charCodeAt(i + 1);
        if (isLowSurrogate(next)) {
          out += input[i] + input[i + 1];
          i++;
          continue;
        }
      }
      out += REPLACEMENT_CHAR;
      continue;
    }
    if (isLowSurrogate(code)) {
      out += REPLACEMENT_CHAR;
      continue;
    }
    out += input[i];
  }
  return out;
}

/** Ensure content is always ContentPart[]. */
export function normalizeContent(content: string | ContentPart[]): ContentPart[] {
  if (typeof content === "string") {
    const safe = sanitizeInvalidSurrogates(content);
    return safe.length > 0 ? [{ type: "text", text: safe }] : [];
  }
  return content
    .map((part) => (part.type === "text" ? { ...part, text: sanitizeInvalidSurrogates(part.text) } : part))
    .filter((part) => part.type !== "text" || part.text.length > 0);
}

/** Extract text-only string from string | ContentPart[]. */
export function contentToString(content: string | ContentPart[]): string {
  if (typeof content === "string") return sanitizeInvalidSurrogates(content);
  return sanitizeInvalidSurrogates(content
    .filter((p): p is Extract<ContentPart, { type: "text" }> => p.type === "text")
    .map((p) => p.text)
    .join(""));
}

/**
 * Filter/degrade content parts based on model's supported input modalities.
 * Unsupported modalities are converted to text placeholders instead of being silently dropped.
 */
export function modalityFilter(
  parts: ContentPart[],
  supported: InputModality[],
): ContentPart[] {
  return parts.map((p) => {
    if (p.type === "file") return p;
    if ((p.type === "text" || p.type === "text_file") && !supported.includes("text")) {
      return { type: "text" as const, text: "[Text: 模型不支持文本输入]" };
    }
    if (p.type === "text" || p.type === "text_file") return p;
    if ((p.type === "image" || p.type === "image_file") && !supported.includes("image")) {
      return { type: "text" as const, text: "[Image: 模型不支持图片输入]" };
    }
    if ((p.type === "video" || p.type === "video_file") && !supported.includes("video")) {
      return { type: "text" as const, text: "[Video: 模型不支持视频输入]" };
    }
    if ((p.type === "audio" || p.type === "audio_file") && !supported.includes("audio")) {
      return { type: "text" as const, text: "[Audio: 模型不支持音频输入]" };
    }
    if (isFileMediaContentPart(p)) return p;
    return p;
  });
}

export function prepareMessagesForModel(
  messages: LLMMessage[],
  supported: InputModality[],
): LLMMessage[] {
  return messages.map((msg) => ({
    ...msg,
    content: modalityFilter(normalizeContent(msg.content), supported),
  }));
}
