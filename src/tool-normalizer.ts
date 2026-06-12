import type { ContentPart } from "@agenteam/types";
import { normalizeContent } from "./modality.js";
import type { LLMMessage, LLMToolCall } from "./types.js";

export interface NormalizedToolTimeline {
  messages: LLMMessage[];
  changed: boolean;
  hasInterruptedToolCalls: boolean;
}

export function isToolErrorResult(result: unknown): boolean {
  return typeof result === "object" && result !== null && "error" in result;
}

export function isTerminalToolMessage(msg: LLMMessage): boolean {
  return msg.role === "tool" && msg.toolStatus !== "pending";
}

export function createPendingToolMessage(toolCall: LLMToolCall): LLMMessage {
  return {
    role: "tool",
    content: normalizeContent(`[Pending: ${toolCall.name}]`),
    toolCallId: toolCall.id,
    toolName: toolCall.name,
    toolStatus: "pending",
    ts: Date.now(),
  };
}

export function createPendingToolMessages(toolCalls: LLMToolCall[]): LLMMessage[] {
  return toolCalls.map(createPendingToolMessage);
}

export function createToolResultMessage(toolCall: LLMToolCall, result: unknown): LLMMessage {
  let content: string | ContentPart[];
  if (isContentPartArray(result)) {
    const { parts, errors } = validateContentParts(result);
    if (errors.length > 0) {
      const errorReport = errors.join("\n");
      content = parts.length > 0
        ? [...parts, { type: "text" as const, text: `\n[ContentPart format error — some parts were dropped]\n${errorReport}` }]
        : `[ContentPart format error — all parts invalid]\n${errorReport}\n\n` +
          `Tool result must use the internal ContentPart format, NOT any LLM provider's API format.\n` +
          `Inline media: { type: "image"|"video"|"audio", data: "<base64>", mimeType: "<mime>" }\n` +
          `File-based:   { type: "image_file"|"video_file"|"audio_file", path: "<path>", mimeType: "<mime>" }\n` +
          `Text:         { type: "text", text: "<content>" }`;
    } else {
      content = parts;
    }
  } else if (typeof result === "string") {
    content = result;
  } else {
    content = JSON.stringify(result);
  }
  return {
    role: "tool",
    content: normalizeContent(content),
    toolCallId: toolCall.id,
    toolName: toolCall.name,
    toolStatus: isToolErrorResult(result) ? "failed" : "completed",
    ts: Date.now(),
  };
}

function isContentPartArray(v: unknown): v is ContentPart[] {
  return Array.isArray(v) && v.length > 0 && typeof (v as any[])[0]?.type === "string";
}

/**
 * Infer required fields from the type name pattern rather than a hardcoded table.
 * Rules:
 *   - "text"                        → needs "text"
 *   - inline media (image/video/audio) → needs "data" + "mimeType"
 *   - anything with _file suffix / "file" → needs "path" + "mimeType"
 *   - unrecognized type              → pass through (no rejection)
 */
const INLINE_MEDIA_TYPES = new Set(["image", "video", "audio"]);

function inferRequiredFields(type: string): string[] | null {
  if (type === "text") return ["text"];
  if (INLINE_MEDIA_TYPES.has(type)) return ["data", "mimeType"];
  if (type === "file" || type === "text_file" || type.endsWith("_file")) return ["path", "mimeType"];
  return null;
}

function validateContentParts(parts: ContentPart[]): { parts: ContentPart[]; errors: string[] } {
  const valid: ContentPart[] = [];
  const errors: string[] = [];
  for (let i = 0; i < parts.length; i++) {
    const p = parts[i] as Record<string, unknown>;
    const t = p.type as string;
    const required = inferRequiredFields(t);
    if (!required) {
      valid.push(parts[i]);
      continue;
    }
    const missing = required.filter((f) => typeof p[f] !== "string");
    if (missing.length > 0) {
      const actualShape = Object.keys(p).map((k) => `${k}: ${typeof p[k]}`).join(", ");
      errors.push(
        `parts[${i}]: type="${t}" requires top-level string fields [${required.join(", ")}], ` +
        `but missing: [${missing.join(", ")}]. Actual shape: { ${actualShape} }`,
      );
      continue;
    }
    valid.push(parts[i]);
  }
  return { parts: valid, errors };
}

export function createSyntheticToolResult(
  toolCall: LLMToolCall,
  reason = `[Error: tool call "${toolCall.name}" was interrupted — no result available]`,
): LLMMessage {
  return {
    role: "tool",
    content: normalizeContent(reason),
    toolCallId: toolCall.id,
    toolName: toolCall.name,
    toolStatus: "synthetic",
  };
}

export function createInterruptedToolResult(
  toolCall: LLMToolCall,
  reason = `[Interrupted: tool call "${toolCall.name}" did not produce a result before the conversation moved on]`,
): LLMMessage {
  return {
    role: "tool",
    content: normalizeContent(reason),
    toolCallId: toolCall.id,
    toolName: toolCall.name,
    toolStatus: "interrupted",
  };
}

/**
 * Normalize tool call / tool result pairing in memory.
 *
 * Pairing rule: every assistant tool_call gets a single matching tool result
 * in the output. WAL-persisted `pending` and missing tool messages are both
 * synthesised as `interrupted` (a synthetic tool_result), because:
 *
 *   buildMessages is the LLM-call entry point, and the agent loop awaits
 *   `runToolBatch` before reaching it — so any `pending` still in the WAL at
 *   read time is a crash artefact (e.g. ENOSPC between writing pending and
 *   writing the real result), never a live in-flight call.
 *
 * Without this, the wire-format converter (`messagesToAnthropic` skips
 * `pending`) emits a `tool_use` block with no matching `tool_result`, which
 * Anthropic rejects with HTTP 400. Synthesising at memory-level keeps the WAL
 * untouched (idempotent: next read fixes it again) while guaranteeing the
 * payload sent on-wire is always paired.
 */
export function normalizeToolTimeline(messages: LLMMessage[]): NormalizedToolTimeline {
  const result: LLMMessage[] = [];
  let hasInterruptedToolCalls = false;
  let changed = false;

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];

    if (msg.role === "tool") {
      changed = true;
      continue;
    }

    result.push(msg);

    if (!msg.toolCalls?.length) {
      continue;
    }

    const bestById = new Map<string, LLMMessage>();
    let cursor = i + 1;

    while (cursor < messages.length && messages[cursor].role === "tool") {
      const toolMsg = messages[cursor];
      const callId = toolMsg.toolCallId;
      if (callId) {
        const normalized = {
          ...toolMsg,
          toolName: toolMsg.toolName ?? msg.toolCalls.find((tc) => tc.id === callId)?.name,
        };
        const current = bestById.get(callId);
        if (!current || compareToolMessagePriority(normalized, current) > 0) {
          bestById.set(callId, normalized);
        }
      }
      cursor++;
    }

    for (const tc of msg.toolCalls) {
      const toolMsg = bestById.get(tc.id);
      if (toolMsg && toolMsg.toolStatus !== "pending") {
        // Real result (completed/failed/synthetic/interrupted) — keep as-is.
        result.push(toolMsg);
        continue;
      }
      // Either no tool message at all, or a `pending` left over from a
      // crashed previous run. Both are unpairable on-wire — synthesise an
      // `interrupted` placeholder so Anthropic / OpenAI / etc. see a valid
      // tool_result alongside the assistant's tool_use.
      result.push(createInterruptedToolResult(tc));
      hasInterruptedToolCalls = true;
      changed = true;
    }

    i = cursor - 1;
  }

  return {
    messages: result,
    changed,
    hasInterruptedToolCalls,
  };
}

function compareToolMessagePriority(a: LLMMessage, b: LLMMessage): number {
  return toolMessagePriority(a) - toolMessagePriority(b);
}

function toolMessagePriority(msg: LLMMessage): number {
  if (msg.role !== "tool") return -1;
  if (msg.toolStatus === "completed" || msg.toolStatus === "failed") {
    return 3;
  }
  if (msg.toolStatus === "synthetic") {
    return 2;
  }
  if (msg.toolStatus === "pending") {
    return 1;
  }
  if (isTerminalToolMessage(msg)) {
    return 3;
  }
  return 0;
}
