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

export interface PrepareInboundMessagesContext {
  signal: AbortSignal;
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

  /** Provider-specific pre-processing hook.
   * Use this for adapter-specific enrichment (e.g. Gemini file refs), not as the generic
   * string -> ContentPart[] normalization entrypoint. */
  prepareInboundMessages?(
    messages: LLMMessage[],
    context: PrepareInboundMessagesContext,
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
