/** Shared utilities for Gemini 2.x and 3.x adapters */

import { createPartFromUri, GoogleGenAI } from "@google/genai";
import type { ContentPart, ToolSchema } from "@agenteam/types";
import { isFileMediaContentPart, isInlineMediaContentPart } from "@agenteam/types";
import type { LLMMessage, ShapeMessagesContext, ShapeCache } from "./types.js";
import type { MediaReaders } from "./media-readers.js";
import { listSupported } from "./types.js";
import { computeSourceKey } from "./shape-cache.js";
import { inlineTextFiles } from "./inline-text-files.js";
import { applyGeminiFilesApi, type GeminiFileRefMap } from "./files-api-helper.js";

const GEMINI_SUPPORTED_IMAGE_MIME_TYPES = new Set([
  "image/heif",
  "image/heic",
  "image/webp",
  "image/jpeg",
  "image/png",
]);
const GEMINI_SUPPORTED_AUDIO_MIME_TYPES = new Set([
  "audio/flac",
  "audio/ogg",
  "audio/aac",
  "audio/aiff",
  "audio/mp3",
  "audio/wav",
]);
const GEMINI_SUPPORTED_VIDEO_MIME_TYPES = new Set([
  "video/3gpp",
  "video/wmv",
  "video/webm",
  "video/mpg",
  "video/x-flv",
  "video/avi",
  "video/mov",
  "video/mpeg",
  "video/mp4",
]);

const GEMINI_SUPPORTED_FILE_MIME_TYPES = new Set([
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document", // .docx
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",       // .xlsx
  "application/vnd.openxmlformats-officedocument.presentationml.presentation", // .pptx
]);

export function describeGeminiError(err: unknown): Record<string, unknown> {
  const read = (value: unknown, depth = 0): Record<string, unknown> | null => {
    if (typeof value !== "object" || value === null || depth > 2) return null;
    const entry = value as Record<string, unknown>;
    return {
      name: typeof entry.name === "string" ? entry.name : undefined,
      message: typeof entry.message === "string" ? entry.message : undefined,
      status: typeof entry.status === "number" ? entry.status : undefined,
      code: typeof entry.code === "string" ? entry.code : undefined,
      stack: typeof entry.stack === "string"
        ? entry.stack.split("\n").slice(0, 3).join("\n")
        : undefined,
      details: Array.isArray(entry.details) ? entry.details : undefined,
      cause: read(entry.cause, depth + 1) ?? undefined,
    };
  };
  return read(err) ?? { value: String(err) };
}

/** Slim record for usageMetadata yield path. Historic `fileRefs` written by
 *  the now-removed inbound `prepareInboundMessages` path are silently ignored
 *  by readers — the new shape pipeline reconstructs Files API URIs at
 *  buildPrompt time, no WAL persistence. Other Google sidecar fields (e.g.
 *  thoughtSignature, thinkingSignature, textSignature) are gemini3-specific
 *  and travel through their own typed channels. */
interface GoogleMessageSidecarRecord {
  usage_raw?: Record<string, unknown>;
}

function sanitizeGeminiSchema(schema: unknown): unknown {
  if (Array.isArray(schema)) {
    return schema.map((item) => sanitizeGeminiSchema(item));
  }
  if (schema == null || typeof schema !== "object") {
    return schema;
  }

  const input = schema as Record<string, unknown>;
  const output: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(input)) {
    output[key] = sanitizeGeminiSchema(value);
  }

  const enumValues = input.enum;
  if (Array.isArray(enumValues) && enumValues.some((value) => typeof value !== "string")) {
    delete output.enum;

    const allowedValues = enumValues.map((value) => JSON.stringify(value)).join(", ");
    const note = `Allowed values: ${allowedValues}.`;
    const description =
      typeof output.description === "string" ? output.description.trim() : "";
    output.description = description ? `${description} ${note}` : note;
  }

  return output;
}

export function toolDefsToGemini(tools?: ToolSchema[]): any[] | undefined {
  if (!tools?.length) return undefined;
  return [
    {
      functionDeclarations: tools.map((t) => ({
        name: t.name,
        description: t.description,
        // Google function schemas reject some JSON Schema variants such as numeric enum values.
        parameters: sanitizeGeminiSchema({
          type: "object",
          properties: t.input_schema.properties,
          required: t.input_schema.required,
        }),
      })),
    },
  ];
}

/**
 * Wire-format conversion. text_file is no longer handled here — the shape
 * pipeline (`inlineTextFiles`) replaces every text_file with a plain text
 * part before this function ever sees the message. Hitting one here is a
 * pipeline bug, not a runtime situation; assertion enforces that contract.
 *
 * `fileRefMap` (closure-scoped, owned by the gemini2/gemini3 provider) maps
 * a part's srcKey → its already-uploaded Files API ref. When a part has a
 * fresh ref the wire-format part becomes `createPartFromUri(...)`, otherwise
 * we fall through to inline encoding (mime-gated by Gemini's supported sets).
 */
export async function contentPartsToGemini(
  content: ContentPart[],
  readers: MediaReaders,
  fileRefMap?: GeminiFileRefMap,
): Promise<any[]> {
  const parts: any[] = [];
  for (const p of content) {
    if (p.type === "text") { parts.push({ text: p.text }); continue; }

    if (p.type === "text_file") {
      // Pipeline invariant: text_file must have been inlined upstream.
      throw new Error(
        `[gemini contentPartsToGemini] received text_file part — shape pipeline ` +
        `did not run inlineTextFiles. Path: ${p.path}`,
      );
    }

    // If a Files API ref is available for this source, prefer it over inline.
    if (fileRefMap) {
      const srcKey = computeSourceKey(p);
      const ref = fileRefMap.get(srcKey);
      if (ref) {
        parts.push(createPartFromUri(ref.uri, ref.mimeType));
        continue;
      }
    }

    if (p.type === "file") {
      let bytes: Buffer;
      let mimeType: string;
      try {
        ({ bytes, mimeType } = await readers.readFileBytes(p));
      } catch {
        parts.push({ text: `[file unavailable: ${p.path}]` });
        continue;
      }
      if (!GEMINI_SUPPORTED_FILE_MIME_TYPES.has(mimeType)) {
        parts.push({ text: `[file unsupported by Gemini: ${p.path} (${mimeType})]` });
        continue;
      }
      parts.push({ inlineData: { data: bytes.toString("base64"), mimeType } });
      continue;
    }

    if (isFileMediaContentPart(p)) {
      try {
        const { bytes, mimeType } = await readers.readMediaBytes(p);
        parts.push({ text: `[file: ${p.path}]` });
        parts.push({ inlineData: { data: bytes.toString("base64"), mimeType } });
      } catch {
        const mediaType = p.type.replace("_file", "");
        parts.push({ text: `[${mediaType} unavailable: ${p.path}]` });
      }
      continue;
    }

    if (!isInlineMediaContentPart(p)) {
      parts.push({ text: `[unknown content type]` });
      continue;
    }

    // Sniff inline bytes via readMediaBytes so dirty mime (e.g. JPEG bytes
    // declared image/png) gets corrected — same hygiene path as file media.
    const { bytes, mimeType: sniffedMime } = await readers.readMediaBytes(p);
    const supportedMimes =
      p.type === "image" ? GEMINI_SUPPORTED_IMAGE_MIME_TYPES
        : p.type === "audio" ? GEMINI_SUPPORTED_AUDIO_MIME_TYPES
          : p.type === "video" ? GEMINI_SUPPORTED_VIDEO_MIME_TYPES
            : null;
    if (supportedMimes && !supportedMimes.has(sniffedMime)) {
      parts.push({
        text:
          `[${p.type}: unsupported by Gemini API for mime ${sniffedMime}. ` +
          `Supported ${p.type} MIME types: ${listSupported(supportedMimes)}]`,
      });
      continue;
    }
    parts.push({ inlineData: { data: bytes.toString("base64"), mimeType: sniffedMime } });
  }
  return parts;
}

/**
 * Gemini's shape pipeline:
 *   1. inlineTextFiles — replace every `text_file` with utf-8 text part.
 *   2. applyGeminiFilesApi — for parts > 20MB, ensure a Files API ref
 *      exists in `fileRefMap` (re-use cached URIs across restarts).
 *
 * Output messages flow into ledger.json + wire only — never WAL.
 */
export async function shapeGeminiMessages(
  client: GoogleGenAI,
  messages: LLMMessage[],
  readers: MediaReaders,
  shapeCache: ShapeCache | undefined,
  fileRefMap: GeminiFileRefMap,
  ctx: ShapeMessagesContext,
): Promise<LLMMessage[]> {
  const inlined = await inlineTextFiles(messages, readers);
  return await applyGeminiFilesApi(
    inlined,
    readers,
    shapeCache,
    fileRefMap,
    client,
    ctx.signal,
  );
}

function toolResultText(msg: LLMMessage): string {
  return msg.content
    .filter((p) => p.type === "text")
    .map((p) => (p as Extract<ContentPart, { type: "text" }>).text)
    .join("");
}

interface ToolResponseCollectOptions {
  textFallback?: boolean;
}

/** Convert a terminal tool result message into a Gemini functionResponse part. */
export function toolResultPartToGemini(msg: LLMMessage): any {
  return {
    functionResponse: {
      name: msg.toolName ?? "unknown_tool",
      response: { result: toolResultText(msg) },
    },
  };
}

/** Group contiguous terminal tool results into one Gemini user turn. */
export function collectToolResponsesToGemini(
  messages: LLMMessage[],
  startIndex: number,
  options: ToolResponseCollectOptions = {},
): {
  content: any;
  nextIndex: number;
} {
  const textFallback = options.textFallback ?? false;
  const parts: any[] = [];
  let index = startIndex;

  while (index < messages.length && messages[index].role === "tool") {
    const msg = messages[index];
    if (msg.toolStatus !== "pending") {
      if (textFallback) {
        const toolName = msg.toolName ?? "unknown_tool";
        const resultText = toolResultText(msg);
        parts.push({
          text: `(Result of ${toolName}: ${(resultText || "(empty)").slice(0, 300)})`,
        });
      } else {
        parts.push(toolResultPartToGemini(msg));
      }
    }
    index++;
  }

  const contentParts =
    parts.length > 0
      ? parts
      : [{ text: "[tool results unavailable]" }];

  return {
    content: {
      role: "user",
      parts: contentParts,
    },
    nextIndex: index,
  };
}

/** Shared usageMetadata → StreamEvent yield logic for Gemini 2.x and 3.x. */
export function* yieldGeminiUsage(chunk: any): Generator<import("./types.js").StreamEvent> {
  const meta = chunk.usageMetadata as Record<string, unknown> | undefined;
  if (!meta) return;
  const sidecarData: GoogleMessageSidecarRecord = { usage_raw: meta };
  yield {
    type: "provider_sidecar",
    providerSidecarData: { google: sidecarData },
  };
  // Outer aggregate: input = promptTokenCount; output = candidates + thoughts (Gemini
  // separates thinking tokens from candidatesTokenCount).
  const promptTokenCount = typeof meta.promptTokenCount === "number" ? meta.promptTokenCount : 0;
  const candidatesTokenCount = typeof meta.candidatesTokenCount === "number" ? meta.candidatesTokenCount : 0;
  const thinkingTokens = typeof meta.thoughtsTokenCount === "number" ? meta.thoughtsTokenCount : 0;
  yield {
    type: "usage",
    inputTokens: promptTokenCount,
    outputTokens: candidatesTokenCount + thinkingTokens,
  };
}

/** Stream response parts and yield StreamEvents. Shared streaming logic. */
export async function* streamGeminiResponse(
  response: AsyncIterable<any>,
  signal: AbortSignal,
): AsyncGenerator<import("./types.js").StreamEvent> {
  let hasContent = false;
  let lastFinishReason: string | undefined;

  for await (const chunk of response) {
    if (signal.aborted) break;

    const candidate = chunk.candidates?.[0];
    if (candidate?.finishReason) {
      lastFinishReason = candidate.finishReason as string;
    }

    const parts = candidate?.content?.parts;
    if (parts) {
      for (const part of parts) {
        if (part.functionCall) {
          hasContent = true;
          const id = `call_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
          yield {
            type: "tool_call",
            id,
            name: part.functionCall.name!,
            arguments: JSON.stringify(part.functionCall.args ?? {}),
          };
        } else if (part.text != null) {
          hasContent = true;
          if ((part as any).thought) {
            yield { type: "thinking", text: part.text };
          } else {
            yield { type: "text", text: part.text };
          }
        }
      }
    }

    if (chunk.usageMetadata) {
      yield* yieldGeminiUsage(chunk);
    }
  }

  if (!hasContent && !signal.aborted && lastFinishReason && lastFinishReason !== "STOP" && lastFinishReason !== "MAX_TOKENS") {
    yield { type: "text", text: `[Gemini blocked: finishReason=${lastFinishReason}]` };
  }
}
