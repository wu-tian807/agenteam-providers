import type { ContentPart, ToolSchema } from "@agenteam/types";
// Message-shape base types now live in @agenteam/types (shared with channels like
// ink-renderer). Engine-only provider plumbing (below) re-imports them from there.
import type {
  ProviderSidecarData,
  SystemBlock,
  LLMMessage,
  LLMToolCall,
} from "@agenteam/types";

export type {
  ProviderSidecarData,
  SystemBlock,
  LLMMessage,
  LLMToolCall,
} from "@agenteam/types";

export function listSupported(values: Iterable<string>): string {
  return Array.from(values).join(", ");
}

/** Flatten the structured `SystemBlock[]` (cache-hint annotated) down to a
 *  single plain-text system prompt — used by adapters whose API does NOT
 *  support cache_control on the system field (openai-compat / openai-response).
 *  The native Anthropic adapter does its own structured serialisation. */
export function blocksToText(blocks: SystemBlock[]): string {
  return blocks.map((b) => b.text).join("\n\n");
}

export type ResponseKind = "text" | "tool_calls" | "thinking_only" | "empty";

export interface LLMResponse {
  content: string | ContentPart[];
  thinking?: string;
  toolCalls?: LLMToolCall[];
  usage?: { inputTokens: number; outputTokens: number };
  /** Framework/catalog model used for this response, not provider-returned version ids. */
  model?: string;
  /** True when the response was cut short by an AbortSignal mid-stream. */
  truncated?: boolean;
  providerSidecarData?: ProviderSidecarData;
  kind: ResponseKind;
}

/** All event types yielded by provider chatStream — internal to the streaming pipeline. */
export type StreamEvent =
  | { type: "text"; text: string; providerSidecarData?: ProviderSidecarData }
  | { type: "thinking"; text: string; providerSidecarData?: ProviderSidecarData }
  | {
      type: "tool_call";
      id: string;
      name: string;
      arguments: string;
      providerSidecarData?: ProviderSidecarData;
    }
  | { type: "provider_sidecar"; providerSidecarData: ProviderSidecarData }
  | { type: "usage"; inputTokens: number; outputTokens: number; model?: string };

export interface ShapeMessagesContext {
  signal: AbortSignal;
}

/** Cache operator the host (src/) injects via ProviderDeps.shapeCache.
 *  Provider helpers (image compression, Gemini Files API) call into this
 *  to persist derived artefacts under {sessionDir}/medias/cache/...
 *  When `shapeCache` is undefined the helpers degrade to in-memory only. */
export interface CachedFilesApiEntry {
  uri: string;
  mimeType: string;
  name?: string;
  /** Absolute timestamp (ms). lookupFilesApi must reject entries
   *  within a 1h safety margin of this value. */
  expiresAt: number;
}

export interface ShapeCache {
  /** Look up a cached compressed image (provider+srcKey+policyHash+ext). */
  lookupImage(opts: {
    providerName: string;
    srcKey: string;
    policyHash: string;
    ext: string;
  }): Promise<{ bytes: Buffer; path: string } | null>;
  writeImage(opts: {
    providerName: string;
    srcKey: string;
    policyHash: string;
    ext: string;
    bytes: Buffer;
  }): Promise<{ path: string }>;
  /** Look up a still-fresh Gemini Files API URI by srcKey. Returns null on miss/expiry. */
  lookupFilesApi(opts: { srcKey: string }): Promise<CachedFilesApiEntry | null>;
  writeFilesApi(opts: {
    srcKey: string;
    entry: CachedFilesApiEntry;
  }): Promise<void>;
}

export interface MaterializeAssistantMessageOptions {
  showThinking: boolean;
  ts: number;
  truncated?: boolean;
}

export interface MaterializeToolMessagesOptions {
  ts: number;
}

export interface LLMProvider {
  chatStream(
    system: SystemBlock[] | undefined,
    messages: LLMMessage[],
    tools: ToolSchema[],
    signal: AbortSignal,
  ): AsyncIterable<StreamEvent>;

  /** Per-buildPrompt message integrity / shaping hook. Runs in the
   *  context-window pipeline AFTER WAL is materialised — output flows
   *  into ledger.json + wire only, never back to events.jsonl.
   *  Each provider composes its own helper chain (e.g. Anthropic does
   *  inlineTextFiles → applyImageCache; Gemini does inlineTextFiles →
   *  applyGeminiFilesApi). */
  shapeMessages?(
    messages: LLMMessage[],
    context: ShapeMessagesContext,
  ): Promise<LLMMessage[]>;

  materializeAssistantMessage?(
    response: LLMResponse,
    options: MaterializeAssistantMessageOptions,
  ): LLMMessage;

  materializePendingToolMessages?(
    toolCalls: LLMToolCall[],
    options: MaterializeToolMessagesOptions,
  ): LLMMessage[];

  /** Returns the tool_result LLMMessage. Synchronous — media hygiene is now
   *  handled at the storage layer (event-blob.ts size-based externalize on
   *  WAL write, media-storage.ts magic-byte mime sniff on file read), not at
   *  this seam. */
  materializeToolResult?(
    toolCall: LLMToolCall,
    result: unknown,
    options: MaterializeToolMessagesOptions,
  ): LLMMessage;
}
