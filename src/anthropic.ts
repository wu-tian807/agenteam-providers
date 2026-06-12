/** Anthropic Messages API adapter — SSE streaming, raw fetch, thinking support */

import { registerProvider, requireApiKey, downgradeEffort, type ProviderFactoryOpts } from "./provider.js";
import type { ContentPart, ToolSchema } from "@agenteam/types";
import type { LLMMessage, LLMProvider, StreamEvent, SystemBlock } from "./types.js";
import { listSupported } from "./types.js";
import { parseSSE } from "./stream.js";
import { annotateLLMError, throwHttpApiError, throwOnStreamError } from "./errors.js";
import type { MediaReaders } from "./media-readers.js";
import {
  base64EncodedSize,
  fitImageToPolicy,
  exceedsImageLimit,
  exceedsImageDimensions,
  describeImageLimit,
  describeImageSize,
  type ImagePreflightPolicy,
} from "./image-compression.js";
import { foldDynamicReminders } from "./dynamic-system.js";

export type AnthropicAssistantBlock =
  | { type: "text"; text: string }
  | { type: "thinking"; thinking: string; signature?: string }
  | { type: "redacted_thinking"; data: string }
  | { type: "tool_use"; id: string; name: string; input: Record<string, unknown> };

interface AnthropicSidecar {
  contentBlocks: AnthropicAssistantBlock[];
  /** Raw Anthropic API usage object — full passthrough, not enumerated. Merged from
   *  `message_start.message.usage` (input/cache/service_tier fields) +
   *  `message_delta.usage` (output_tokens delta). New API fields automatically captured.
   *  Schema: see Anthropic Messages API docs (input_tokens, output_tokens,
   *  cache_read_input_tokens, cache_creation_input_tokens, cache_creation,
   *  service_tier, …).
   *  Named `usage_raw` (not `usage`) to avoid collision with the framework-level
   *  outer `usage: { inputTokens, outputTokens }` standardized aggregate. */
  usage_raw?: Record<string, unknown>;
}

/**
 * Detect models that require adaptive-only thinking and reject sampling parameters.
 * Matches claude-opus-4-7, claude-opus-4-7-20260416, and any future opus >= 4.7.
 */
export function isAdaptiveOnlyModel(model: string): boolean {
  const match = model.match(/claude-opus-(\d+)-(\d+)/);
  if (!match) return false;
  const major = parseInt(match[1], 10);
  const minor = parseInt(match[2], 10);
  return major > 4 || (major === 4 && minor >= 7);
}

/**
 * Detect models that support mid-conversation native system messages.
 * Per Anthropic docs: "This feature is available on Claude Opus 4.8 only."
 * Matches claude-opus-4-8 and its dated variants (e.g. claude-opus-4-8-20260519).
 * Do NOT extend to future models without verifying support.
 */
export function supportsMidConvoSystem(model: string): boolean {
  return /^claude-opus-4-8\b/i.test(model);
}

/**
 * Detect models that support Anthropic fast mode (research preview).
 * Per Anthropic docs: claude-opus-4-6, 4-7, and 4-8 support fast mode.
 * Explicit allowlist — do NOT extend to future models without verifying support.
 */
export function supportsFastMode(model: string): boolean {
  return /^claude-opus-4-(?:6|7|8)\b/i.test(model);
}

const ANTHROPIC_SUPPORTED_IMAGE_MIME_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
]);
const ANTHROPIC_IMAGE_MAX_BASE64_BYTES = 5 * 1024 * 1024;

const ANTHROPIC_SUPPORTED_DOCUMENT_MIME_TYPES = new Set([
  "application/pdf",
]);

const IMAGE_POLICY: ImagePreflightPolicy = {
  maxBase64Bytes: ANTHROPIC_IMAGE_MAX_BASE64_BYTES,
  maxLongEdge: 2000,
  compressOversized: true,
  supportedMimeTypes: ANTHROPIC_SUPPORTED_IMAGE_MIME_TYPES,
};

export function toolDefsToAnthropic(tools?: ToolSchema[]) {
  if (!tools?.length) return undefined;
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: t.input_schema,
  }));
}

async function contentToAnthropic(content: ContentPart[], readers: MediaReaders): Promise<any[]> {
  const result: any[] = [];
  for (const p of content) {
    const converted = await convertPartToAnthropic(p, readers);
    if (Array.isArray(converted)) {
      result.push(...converted);
    } else {
      result.push(converted);
    }
  }
  return result;
}

async function convertPartToAnthropic(p: ContentPart, readers: MediaReaders): Promise<any | any[]> {
  if (p.type === "text") {
    if (p.text.length === 0) return [];
    return { type: "text", text: p.text };
  }

  if (p.type === "text_file") {
    // text_file is always inlined as a text block (path label + UTF-8 body),
    // never routed through the native `document` channel. The deployed
    // Anthropic backend is Bedrock-Anthropic (via the forgeax proxy), whose
    // `document.source.base64.media_type` is a single-value enum that only
    // accepts `application/pdf` — `text/plain` / `text/markdown` documents are
    // rejected outright. Inlining also costs no base64 inflation and works
    // uniformly across every backend. Native `document` is reserved for PDF
    // (see the `file` branch below, gated on ANTHROPIC_SUPPORTED_DOCUMENT_MIME_TYPES).
    try {
      const { bytes } = await readers.readFileBytes(p);
      return { type: "text", text: `[file: ${p.path}]\n${bytes.toString("utf8")}` };
    } catch {
      return { type: "text", text: `[file unavailable: ${p.path}]` };
    }
  }

  if (p.type === "file") {
    // Read first to let readFileBytes sniff-correct mime; gate uses corrected
    // mime so PDF bytes declared application/octet-stream still get routed to
    // the document path instead of being rejected by a declared-mime gate.
    let bytes: Buffer;
    let mimeType: string;
    try {
      ({ bytes, mimeType } = await readers.readFileBytes(p));
    } catch {
      return { type: "text", text: `[file unavailable: ${p.path}]` };
    }
    if (!ANTHROPIC_SUPPORTED_DOCUMENT_MIME_TYPES.has(mimeType)) {
      return { type: "text", text: `[file unsupported by Anthropic: ${p.path} (${mimeType})]` };
    }
    return [
      { type: "text", text: `[file: ${p.path}]` },
      { type: "document", source: { type: "base64", media_type: mimeType, data: bytes.toString("base64") } },
    ];
  }

  if (p.type === "image_file") {
    try {
      const loaded = await readers.readMediaBytes(p);
      const img = await convertInlineImageToAnthropic(loaded.bytes, loaded.mimeType, loaded.label);
      return [{ type: "text", text: `[file: ${p.path}]` }, img];
    } catch {
      return { type: "text", text: `[image unavailable: ${p.path}]` };
    }
  }

  if (p.type === "audio_file" || p.type === "video_file") {
    const mediaType = p.type.replace("_file", "");
    return { type: "text", text: `[${mediaType} unsupported by Anthropic: ${p.path}]` };
  }

  if (p.type === "image") {
    // Use readMediaBytes so inline mime gets sniff-corrected (single dispatch
    // point for media bytes — same hygiene path as image_file).
    const { bytes, mimeType } = await readers.readMediaBytes(p);
    return await convertInlineImageToAnthropic(bytes, mimeType, `inline image`);
  }

  if (p.type === "video" || p.type === "audio") {
    return {
      type: "text",
      text:
        `[${p.type}: unsupported by Anthropic messages API. ` +
        `This API currently supports image MIME types ${listSupported(ANTHROPIC_SUPPORTED_IMAGE_MIME_TYPES)}]`,
    };
  }

  return { type: "text", text: `[unknown content type]` };
}

async function convertInlineImageToAnthropic(
  bytes: Buffer,
  mimeType: string,
  label: string,
): Promise<any> {
  if (!ANTHROPIC_SUPPORTED_IMAGE_MIME_TYPES.has(mimeType)) {
    return {
      type: "text",
      text:
        `[image: unsupported by Anthropic messages API for mime ${mimeType}. ` +
        `Supported image MIME types: ${listSupported(ANTHROPIC_SUPPORTED_IMAGE_MIME_TYPES)}]`,
    };
  }

  if (!exceedsImageLimit(bytes, IMAGE_POLICY) && !(await exceedsImageDimensions(bytes, IMAGE_POLICY))) {
    return {
      type: "image",
      source: { type: "base64", media_type: mimeType, data: bytes.toString("base64") },
    };
  }

  if (IMAGE_POLICY.compressOversized) {
    const normalized = await fitImageToPolicy(bytes, mimeType, IMAGE_POLICY);
    if (normalized && !exceedsImageLimit(normalized.bytes, IMAGE_POLICY)) {
      return {
        type: "image",
        source: { type: "base64", media_type: normalized.mimeType, data: normalized.bytes.toString("base64") },
      };
    }
  }

  return {
    type: "text",
    text:
      `[image omitted: Anthropic image limit is ${describeImageLimit(IMAGE_POLICY)} ` +
      `and ${label} is ${describeImageSize(bytes)}]`,
  };
}

function toAnthropicSidecar(sidecarData: LLMMessage["providerSidecarData"]): AnthropicSidecar | null {
  const raw = sidecarData?.anthropic;
  if (!raw || typeof raw !== "object") return null;
  const contentBlocks = (raw as { contentBlocks?: unknown }).contentBlocks;
  if (!Array.isArray(contentBlocks)) return null;

  const normalized = contentBlocks.flatMap((block): AnthropicAssistantBlock[] => {
    if (!block || typeof block !== "object") return [];
    const item = block as Record<string, unknown>;
    switch (item.type) {
      case "text":
        return typeof item.text === "string" && item.text.length > 0
          ? [{ type: "text", text: item.text }]
          : [];
      case "thinking":
        return typeof item.thinking === "string"
          ? [{
              type: "thinking",
              thinking: item.thinking,
              signature: typeof item.signature === "string" ? item.signature : undefined,
            }]
          : [];
      case "redacted_thinking":
        return typeof item.data === "string"
          ? [{ type: "redacted_thinking", data: item.data }]
          : [];
      case "tool_use":
        return typeof item.id === "string" &&
          typeof item.name === "string" &&
          item.input &&
          typeof item.input === "object" &&
          !Array.isArray(item.input)
          ? [{
              type: "tool_use",
              id: item.id,
              name: item.name,
              input: item.input as Record<string, unknown>,
            }]
          : [];
      default:
        return [];
    }
  });

  return normalized.length > 0 ? { contentBlocks: normalized } : null;
}

function anthropicBlocksToApi(blocks: AnthropicAssistantBlock[]): any[] {
  return blocks.map((block) => {
    switch (block.type) {
      case "thinking":
        return {
          type: "thinking",
          thinking: block.thinking,
          ...(block.signature ? { signature: block.signature } : {}),
        };
      case "redacted_thinking":
        return { type: "redacted_thinking", data: block.data };
      default:
        return block;
    }
  });
}

async function fallbackAssistantBlocks(msg: LLMMessage, readers: MediaReaders): Promise<any[]> {
  const blocks: any[] = [];
  if (msg.content) {
    const c = await contentToAnthropic(msg.content, readers);
    blocks.push(...c);
  }
  for (const tc of msg.toolCalls ?? []) {
    blocks.push({
      type: "tool_use",
      id: tc.id,
      name: tc.name,
      input: tc.arguments,
    });
  }
  return blocks;
}

async function assistantMessageToAnthropicContent(msg: LLMMessage, readers: MediaReaders): Promise<any[]> {
  const sidecar = toAnthropicSidecar(msg.providerSidecarData);
  if (sidecar?.contentBlocks.length) {
    return anthropicBlocksToApi(sidecar.contentBlocks);
  }
  return fallbackAssistantBlocks(msg, readers);
}

async function messagesToAnthropic(messages: LLMMessage[], readers: MediaReaders): Promise<any[]> {
  const result: any[] = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];

    if (msg.role === "tool") {
      const toolBlocks: any[] = [];
      let j = i;
      while (j < messages.length && messages[j].role === "tool") {
        const toolMsg = messages[j];
        // SSOT: `tool-normalizer.ts:normalizeToolTimeline` is the single owner of
        // toolStatus invariants — by the time messages reach this wire converter,
        // every `tool` role message MUST be a real result (completed/failed/
        // synthetic/interrupted), never `pending`. Surfacing pending here would
        // leak a `tool_use` block with no matching `tool_result` and Anthropic
        // would reject the whole turn with HTTP 400. Fail loud instead of
        // silently skipping (which masks the upstream contract violation).
        if (toolMsg.toolStatus === "pending") {
          throw new Error(
            `messagesToAnthropic invariant violation: encountered pending tool message ` +
            `(toolCallId=${toolMsg.toolCallId ?? "<none>"}). normalizeToolTimeline must run ` +
            `before wire conversion — see tool-normalizer.ts contract.`,
          );
        }
        toolBlocks.push({
          type: "tool_result",
          tool_use_id: toolMsg.toolCallId ?? "_tool",
          content: await contentToAnthropic(toolMsg.content, readers),
        });
        j++;
      }
      if (toolBlocks.length > 0) {
        result.push({
          role: "user",
          content: toolBlocks,
        });
      }
      i = j - 1;
      continue;
    }

    if (msg.role === "assistant") {
      const blocks = await assistantMessageToAnthropicContent(msg, readers);
      if (blocks.length > 0) {
        result.push({ role: "assistant", content: blocks });
      }
      continue;
    }

    if (msg.role === "system") {
      // Opus 4.8 native mid-conversation system message. Carrier content is the
      // raw dynamic block text (no <system-reminder> wrapper — the model treats
      // it as genuine system context). Only reached on the native path; the fold
      // path strips role:"system" before conversion.
      //
      // Anthropic requires a mid-convo system block to follow a user turn
      // (tool_result batches convert to role:"user", so they count). Placement
      // depends only on the previously emitted wire message:
      //   - prev is user   → emit native system (valid).
      //   - prev is system → merge into it (consecutive carriers collapse to one
      //                       block that still follows the earlier user turn).
      //   - otherwise (assistant tail / history head) → degrade THIS carrier to
      //                       a role:"user" message so it stays valid.
      const sysContent = await contentToAnthropic(msg.content, readers);
      const prev = result[result.length - 1];
      if (prev && prev.role === "system") {
        prev.content = [...prev.content, ...sysContent];
      } else if (prev && prev.role === "user") {
        result.push({ role: "system", content: sysContent });
      } else {
        result.push({ role: "user", content: sysContent });
      }
      continue;
    }

    result.push({
      role: "user",
      content: await contentToAnthropic(msg.content, readers),
    });
  }
  return result;
}

export function systemBlocksToAnthropic(blocks: SystemBlock[]): any[] {
  return blocks.map((block, i, arr) => {
    const entry: any = { type: "text", text: block.text };
    const nextIsNotStable = i === arr.length - 1 || arr[i + 1].cacheHint !== "stable";
    if (block.cacheHint === "stable" && nextIsNotStable) {
      entry.cache_control = { type: "ephemeral" };
    }
    return entry;
  });
}

export function annotateMessageCache(messages: any[]): void {
  // Anchor the cache marker to the LAST user turn on-wire. The carrier
  // architecture (history-pipeline + dynamic-system) makes every user / tool
  // turn — including the one that triggered this loop — byte-stable across
  // replays: the dynamic delta lives in its own immutable `role:"system"`
  // carrier event, and providers either keep it native (Opus 4.8) or fold it
  // into the preceding immutable turn (everything else). Either way, the
  // current call's user / tool_result block is identical to what will be
  // replayed next time we land on this same prefix → marking the last user
  // turn caches one extra round-trip's worth of context (the previous
  // assistant turn + the current user input / tool_result batch) compared to
  // marking second-to-last. The earlier two-marker dance was a workaround
  // for the legacy in-message reminder splice, which mutated the last user
  // turn and broke prefix stability — no longer applicable.
  let lastUserIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") { lastUserIdx = i; break; }
  }
  if (lastUserIdx < 0) return;
  const content = messages[lastUserIdx].content;
  if (!Array.isArray(content) || content.length === 0) return;
  // Skip trailing <system-reminder> blocks (folded reminders can sit at the
  // content tail) so the marker lands on stable, real content.
  let markerIdx = content.length - 1;
  while (
    markerIdx >= 0 &&
    content[markerIdx]?.type === "text" &&
    typeof content[markerIdx]?.text === "string" &&
    content[markerIdx].text.startsWith("<system-reminder>")
  ) {
    markerIdx--;
  }
  if (markerIdx < 0) return;
  content[markerIdx] = {
    ...content[markerIdx],
    cache_control: { type: "ephemeral" },
  };
}

/**
 * Consume an Anthropic-format event stream and yield framework `StreamEvent`s.
 *
 * The input `events` async iterable yields `{event, data}` where `data` is
 * ALREADY JSON-parsed. Native Anthropic (`/v1/messages`) wraps each chunk in a
 * text/event-stream frame and feeds raw strings via `parseSSE`; Bedrock
 * (`invoke-with-response-stream`) wraps them in AWS event-stream binary frames
 * and decodes via `parseAwsEventStream`. Both converge on the SAME
 * `{event, data}` post-parse shape — this helper is what they share.
 *
 * `providerLabel` (e.g. "anthropic" or "aws-bedrock-anthropic") feeds error
 * annotation only; it does not change wire behavior.
 */
export async function* consumeAnthropicEvents(
  events: AsyncIterable<{ event?: string; data: any }>,
  signal: AbortSignal,
  providerLabel: string,
  model: string,
): AsyncGenerator<StreamEvent, void, void> {
  let currentBlock:
    | { type: "text"; text: string }
    | { type: "thinking"; thinking: string; signature?: string }
    | { type: "redacted_thinking"; data: string }
    | { type: "tool_use"; id?: string; name?: string; arguments: string }
    | null = null;
  const assistantBlocks: AnthropicAssistantBlock[] = [];
  let outputTokens = 0;
  let rawUsage: Record<string, unknown> | undefined;

  for await (const { event, data: parsed } of events) {
    if (signal.aborted) break;
    if (!parsed) continue;

    throwOnStreamError(parsed, providerLabel, model);

    const eventType = event ?? parsed.type;

    switch (eventType) {
      case "message_start": {
        if (parsed.message?.usage && typeof parsed.message.usage === "object") {
          rawUsage = { ...(parsed.message.usage as Record<string, unknown>) };
        }
        break;
      }
      case "content_block_start": {
        const block = parsed.content_block;
        if (block?.type === "tool_use") {
          currentBlock = { type: "tool_use", id: block.id, name: block.name, arguments: "" };
        } else if (block?.type === "thinking") {
          currentBlock = {
            type: "thinking",
            thinking: typeof block.thinking === "string" ? block.thinking : "",
            signature: typeof block.signature === "string" ? block.signature : undefined,
          };
        } else if (block?.type === "redacted_thinking") {
          currentBlock = {
            type: "redacted_thinking",
            data: typeof block.data === "string" ? block.data : "",
          };
        } else {
          currentBlock = {
            type: "text",
            text: typeof block?.text === "string" ? block.text : "",
          };
        }
        break;
      }
      case "content_block_delta": {
        const delta = parsed.delta;
        if (delta?.type === "text_delta" && delta.text) {
          if (currentBlock?.type === "text") currentBlock.text += delta.text;
          yield { type: "text", text: delta.text };
        } else if (delta?.type === "thinking_delta" && delta.thinking) {
          if (currentBlock?.type === "thinking") currentBlock.thinking += delta.thinking;
          yield { type: "thinking", text: delta.thinking };
        } else if (delta?.type === "signature_delta" && currentBlock?.type === "thinking") {
          currentBlock.signature = delta.signature;
        } else if (
          delta?.type === "input_json_delta" &&
          delta.partial_json &&
          currentBlock
        ) {
          if (currentBlock.type === "tool_use") {
            currentBlock.arguments += delta.partial_json;
          }
        }
        break;
      }
      case "content_block_stop":
        if (currentBlock?.type === "tool_use") {
          let parsedArguments: Record<string, unknown> = {};
          try {
            parsedArguments = currentBlock.arguments ? JSON.parse(currentBlock.arguments) : {};
          } catch { /* keep best-effort empty obj */ }
          assistantBlocks.push({
            type: "tool_use",
            id: currentBlock.id ?? "_tool",
            name: currentBlock.name ?? "unknown_tool",
            input: parsedArguments,
          });
          yield {
            type: "tool_call",
            id: currentBlock.id ?? "_tool",
            name: currentBlock.name ?? "unknown_tool",
            arguments: currentBlock.arguments,
          };
        } else if (currentBlock?.type === "thinking") {
          assistantBlocks.push({
            type: "thinking",
            thinking: currentBlock.thinking,
            ...(currentBlock.signature ? { signature: currentBlock.signature } : {}),
          });
        } else if (currentBlock?.type === "redacted_thinking") {
          assistantBlocks.push(currentBlock);
        } else if (currentBlock?.type === "text" && currentBlock.text) {
          assistantBlocks.push(currentBlock);
        }
        currentBlock = null;
        break;
      case "message_delta":
        if (parsed.usage && typeof parsed.usage === "object") {
          rawUsage = { ...(rawUsage ?? {}), ...(parsed.usage as Record<string, unknown>) };
          if (typeof parsed.usage.output_tokens === "number") {
            outputTokens = parsed.usage.output_tokens;
          }
        }
        break;
      case "message_stop": {
        if (assistantBlocks.length > 0 || rawUsage) {
          const anthropicData: AnthropicSidecar = {
            contentBlocks: assistantBlocks,
            ...(rawUsage ? { usage_raw: rawUsage } : {}),
          };
          yield {
            type: "provider_sidecar",
            providerSidecarData: { anthropic: anthropicData },
          };
        }
        const nakedInput = typeof rawUsage?.input_tokens === "number" ? rawUsage.input_tokens : 0;
        const cacheRead = typeof rawUsage?.cache_read_input_tokens === "number" ? rawUsage.cache_read_input_tokens : 0;
        const cacheCreate = typeof rawUsage?.cache_creation_input_tokens === "number" ? rawUsage.cache_creation_input_tokens : 0;
        yield { type: "usage", inputTokens: nakedInput + cacheRead + cacheCreate, outputTokens };
        break;
      }
    }
  }
}

function createAnthropicProvider(opts: ProviderFactoryOpts): LLMProvider {
  const { apiKey, baseUrl, readers } = opts;
  const base = (baseUrl ?? "https://api.anthropic.com").replace(/\/+$/, "");
  const url = `${base}/v1/messages`;
  const model = opts.model;
  const temperature = opts.temperature ?? 0.7;
  const maxTokens = opts.maxTokens ?? 4096;
  const reasoningEffort = opts.reasoningEffort;
  const fast = opts.fast;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01",
    "x-api-key": apiKey,
  };
  // Fast mode: Opus 4.6/4.7/4.8 support accelerated inference via beta header
  if (fast && supportsFastMode(model)) {
    headers["anthropic-beta"] = "fast-mode-2026-02-01";
  }

  return {
    async prepareInboundMessages(messages, _context) {
      return messages;
    },
    async *chatStream(system, messages, tools, signal) {
      // `system` is stable-only → the `system` field. Dynamic blocks ride in
      // history as immutable role:"system" carriers.
      //   - Opus 4.8 (supportsMidConvoSystem): keep the carriers and let
      //     messagesToAnthropic emit native {role:"system"} messages. Each
      //     carrier is anchored right after the user / tool_result that
      //     triggered the loop, so on-wire it follows a `user` turn (a real
      //     user message, or a tool_result batch converted to role:"user") —
      //     satisfying Anthropic's "system must follow a user turn" constraint.
      //   - Other models: fold the carriers (two-layer smoosh) into their
      //     preceding immutable turn.
      // Either way the carrier bytes are immutable → cache prefix stays warm.
      const stable = system ?? [];
      const useNative = supportsMidConvoSystem(model);

      const anthropicMessages = await messagesToAnthropic(
        useNative ? messages : foldDynamicReminders(messages),
        readers,
      );
      const anthropicTools = toolDefsToAnthropic(tools);

      const body: any = {
        model,
        messages: anthropicMessages,
        max_tokens: maxTokens,
        stream: true,
      };
      if (stable.length) {
        body.system = systemBlocksToAnthropic(stable);
      }
      annotateMessageCache(anthropicMessages);
      if (anthropicTools) body.tools = anthropicTools;

      if (isAdaptiveOnlyModel(model)) {
        // Opus 4.7+: adaptive thinking only, no sampling parameters
        if (reasoningEffort) {
          body.thinking = { type: "adaptive", display: "summarized" };
          body.output_config = { effort: reasoningEffort };
        }
        // No temperature/top_p/top_k — 4.7 rejects non-default values
      } else {
        // Opus 4.6 and below: budget-based thinking + temperature
        if (reasoningEffort) {
          const effective = downgradeEffort(reasoningEffort, ["low", "medium", "high"]);
          const budget = effective === "high" ? 32768 : effective === "medium" ? 16384 : 8192;
          body.thinking = { type: "enabled", budget_tokens: budget };
          // Fail-safe: max_tokens must be > budget_tokens or Anthropic rejects (HTTP 400).
          // Caller is expected to size max_tokens to fit; this guards against forgetting.
          if ((body.max_tokens as number) <= budget) {
            body.max_tokens = budget + 4096;
          }
          body.temperature = 1; // required when thinking is enabled
        } else {
          body.temperature = temperature;
        }
      }

      // Fast mode: body.speed independent of thinking/sampling path (supports 4.6/4.7/4.8)
      if (fast && supportsFastMode(model)) {
        body.speed = "fast";
      }

      const res = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal,
      });
      if (!res.ok) {
        const text = await res.text();
        throwHttpApiError(res, text, "anthropic", model);
      }

      // Adapt parseSSE's `{event?, data: string}` shape to the parsed shape
      // consumeAnthropicEvents wants. Malformed JSON chunks (rare; usually
      // keep-alive heartbeats with empty data) are silently dropped, matching
      // the previous inline behavior.
      async function* parsedEvents() {
        for await (const { event, data } of parseSSE(res)) {
          let parsed: any;
          try { parsed = JSON.parse(data); } catch { continue; }
          yield { event, data: parsed };
        }
      }
      yield* consumeAnthropicEvents(parsedEvents(), signal, "anthropic", model);
    },
  };
}

registerProvider("anthropic-messages", createAnthropicProvider, {
  validateKey: requireApiKey,
});

export { createAnthropicProvider, contentToAnthropic, messagesToAnthropic };
