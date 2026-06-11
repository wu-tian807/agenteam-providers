/**
 * Claude-via-OpenAI-Compatible adapter
 * =====================================
 *
 * Use Anthropic Claude models through an OpenAI-compatible endpoint
 * (LiteLLM / forgeax / OpenRouter / corporate gateway) while keeping
 * the cache-control + signed-thinking guarantees that the native
 * Anthropic adapter provides.
 *
 * Why this exists
 * ---------------
 * Some LiteLLM proxies (forgeax in particular) have a streaming bug in
 * their `/v1/messages` (anthropic-style) endpoint: when extended thinking
 * is enabled the SSE aggregator misroutes `text_delta` events into the
 * thinking block, so the final assistant text comes back mostly empty.
 * Switching the same model to the `/v1/chat/completions` (openai-style)
 * endpoint avoids the bug entirely — content and reasoning travel on
 * separate fields (`delta.content` vs `delta.reasoning_content`) so there
 * is nothing to misroute.
 *
 * The bare `openai-completions` provider however drops three things that
 * production Claude usage depends on:
 *
 *   1. **Prompt cache** — its `system` is serialised as a plain string
 *      with no `cache_control` markers. LiteLLM does NOT add implicit
 *      caching for Anthropic backends, so every request pays full
 *      price for the system prompt and the conversation prefix.
 *   2. **Signed thinking blocks** — extended-thinking responses carry a
 *      `signature` field on each thinking block. Bedrock-Anthropic
 *      rejects multi-turn tool-use whose history omits the signature.
 *      The vanilla adapter never reads `delta.thinking_blocks`, so the
 *      signature is silently discarded.
 *   3. **Precise thinking budget** — `reasoning_effort: low/medium/high`
 *      is mapped server-side to opaque budgets (and is not even
 *      monotonic on some proxies). The native adapter sends an explicit
 *      `thinking: {budget_tokens: N}` instead.
 *
 * This adapter reuses the openai-compat message-shape encoder (so
 * tool-calls, multimodal inputs and provider sidecar plumbing stay
 * identical) and adds the three missing pieces on top.
 *
 * Design notes
 * ------------
 *  - System messages: structured `[{type:"text", text, cache_control?}]`
 *    content with `cache_control: {type: "ephemeral"}` placed on the
 *    last stable block. Mirrors `anthropic.ts:systemBlocksToAnthropic`.
 *  - Conversation prefix: a second cache marker is dropped on the
 *    second-to-last user message, skipping trailing `<system-reminder>`
 *    blocks. Mirrors `anthropic.ts:annotateMessageCache`.
 *  - Thinking: when `reasoningEffort` is set we send the Anthropic
 *    native `thinking: {type:"enabled", budget_tokens: N}` as an
 *    extra top-level field (LiteLLM and forgeax both pass this through
 *    to the upstream Anthropic API). This also gives us the same cache
 *    namespace as the native `/v1/messages` adapter, so existing caches
 *    transfer.
 *  - Signed history replay: thinking blocks (with `signature`) captured
 *    during streaming are written into a provider sidecar and replayed
 *    on the assistant message during subsequent turns via an
 *    `assistantTransform` hook on `messagesToOpenAI`.
 *  - Opus 4.7+ adaptive thinking: handled with `thinking: {type:
 *    "adaptive"}` and `output_config.effort` exactly like the native
 *    adapter — sampling parameters are also dropped on these models.
 *
 * What we deliberately do NOT do here
 * -----------------------------------
 *  - Do not modify `openai-compat.ts`. Other openai-compat providers
 *    (GPT-5, Qwen, ...) reject `cache_control` / `thinking_blocks`
 *    fields and must stay clean.
 *  - Do not forward `reasoning_effort` when `thinking` is set. LiteLLM
 *    treats them as separate routing paths with separate cache
 *    namespaces; sending both causes the upstream to ignore one and
 *    can split your cache hit rate in half.
 */

import { registerProvider, requireApiKey, downgradeEffort, type ProviderFactoryOpts } from "./provider.js";
import type { ReasoningEffort } from "@agenteam/types";
import type { LLMProvider, StreamEvent, SystemBlock, LLMMessage } from "./types.js";
import { parseSSE } from "./stream.js";
import { throwHttpApiError, throwOnStreamError } from "./errors.js";
import {
  toolDefsToOpenAI,
  messagesToOpenAI,
  normalizeBaseUrl,
  type OpenAIAssistantTransform,
} from "./openai-compat.js";
import { foldDynamicReminders } from "./dynamic-system.js";
import { inlineTextFiles } from "./inline-text-files.js";

// ── Sidecar ──────────────────────────────────────────────────────

/**
 * Stored on assistant messages produced by this adapter. The streaming layer
 * writes `usage_raw` and `thinking_blocks`; the assistantTransform reads
 * `thinking_blocks` on subsequent turns to replay signed thinking history
 * (Bedrock-Anthropic rejects multi-turn tool_use whose preceding thinking
 * blocks are unsigned).
 */
interface ClaudeOpenAICompatSidecar {
  /** Raw OpenAI-compat usage object (prompt_tokens_details.cached_tokens,
   *  cache_read_input_tokens, cache_creation_input_tokens, ...). Captured
   *  verbatim so downstream analytics can read new fields without changes. */
  usage_raw?: Record<string, unknown>;
  /** Fully-assembled thinking blocks in stream order. `signature` is required
   *  by Bedrock when these blocks precede a tool_use; we drop unsigned
   *  blocks on replay (see `assistantTransform`) rather than send an
   *  unsigned shape that the upstream will reject. */
  thinking_blocks?: Array<{
    type: "thinking" | "redacted_thinking";
    thinking?: string;
    signature?: string;
    data?: string;
  }>;
}

// ── System / cache marker construction ────────────────────────────

/**
 * Build the structured content array for the `system` message and place
 * cache_control on the last block of each "stable" run. Mirrors
 * `anthropic.ts:systemBlocksToAnthropic`.
 */
function systemBlocksToOpenAIClaude(blocks: SystemBlock[]): Array<Record<string, unknown>> {
  return blocks.map((block, i, arr) => {
    const entry: Record<string, unknown> = { type: "text", text: block.text };
    const nextIsNotStable = i === arr.length - 1 || arr[i + 1].cacheHint !== "stable";
    if (block.cacheHint === "stable" && nextIsNotStable) {
      entry.cache_control = { type: "ephemeral" };
    }
    return entry;
  });
}

/**
 * Drop a cache_control marker on the LAST user message so the conversation
 * prefix is cache-eligible. The carrier architecture (history-pipeline +
 * dynamic-system) keeps every user / tool turn byte-stable across replays —
 * dynamic deltas live in their own immutable `role:"system"` carrier event,
 * folded into the preceding turn — so anchoring on the last user turn caches
 * one extra round-trip's worth of context vs. second-to-last. The earlier
 * two-marker dance was a workaround for the legacy in-message reminder
 * splice that mutated the tail user turn; no longer applicable. Trailing
 * `<system-reminder>` blocks are still skipped so the marker lands on stable
 * real content. Mirrors `anthropic.ts:annotateMessageCache`, ported to
 * OpenAI-style content arrays.
 */
function annotateMessageCacheOpenAI(messages: Array<Record<string, any>>): void {
  let lastUserIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") { lastUserIdx = i; break; }
  }
  if (lastUserIdx < 0) return;
  const content = messages[lastUserIdx].content;
  if (!Array.isArray(content) || content.length === 0) return;

  let markerIdx = content.length - 1;
  while (
    markerIdx >= 0 &&
    content[markerIdx]?.type === "text" &&
    typeof content[markerIdx].text === "string" &&
    (content[markerIdx].text as string).startsWith("<system-reminder>")
  ) {
    markerIdx--;
  }
  if (markerIdx < 0) return;
  content[markerIdx] = {
    ...content[markerIdx],
    cache_control: { type: "ephemeral" },
  };
}

// ── Thinking config ──────────────────────────────────────────────

/**
 * Opus 4.7+ rejects sampling parameters (temperature/top_p/top_k) and only
 * accepts `thinking: {type: "adaptive"}` instead of explicit budget. Mirrors
 * the same gate in `anthropic.ts:isAdaptiveOnlyModel`.
 */
function isAdaptiveOnlyModel(model: string): boolean {
  const match = model.match(/claude-opus-(\d+)-(\d+)/);
  if (!match) return false;
  const major = parseInt(match[1], 10);
  const minor = parseInt(match[2], 10);
  return major > 4 || (major === 4 && minor >= 7);
}

function effortToBudget(eff: "low" | "medium" | "high"): number {
  return eff === "high" ? 32768 : eff === "medium" ? 16384 : 8192;
}

// ── Thinking-blocks stream accumulator ───────────────────────────

/**
 * LiteLLM's openai-compat output for Anthropic models emits thinking deltas
 * as `delta.thinking_blocks: [{type, thinking?, signature?, data?}]`. The
 * text arrives in repeated entries with the `thinking` field set, then a
 * single closing entry carries `signature`. Redacted blocks arrive as a
 * one-shot entry with `data`.
 *
 * We accumulate per-block (text into a single open block until its signature
 * arrives) so the replayed history matches the original block structure.
 */
class ThinkingBlocksAccumulator {
  private finalized: NonNullable<ClaudeOpenAICompatSidecar["thinking_blocks"]> = [];
  private open: {
    type: "thinking";
    thinking: string;
    signature?: string;
  } | null = null;

  ingest(entries: unknown[]): void {
    for (const raw of entries) {
      if (!raw || typeof raw !== "object") continue;
      const entry = raw as Record<string, unknown>;

      if (typeof entry.data === "string" && entry.data.length > 0) {
        if (this.open) {
          this.finalized.push(this.open);
          this.open = null;
        }
        this.finalized.push({ type: "redacted_thinking", data: entry.data });
        continue;
      }

      if (typeof entry.thinking === "string" && entry.thinking.length > 0) {
        if (!this.open) this.open = { type: "thinking", thinking: "" };
        this.open.thinking += entry.thinking;
      }

      if (typeof entry.signature === "string" && entry.signature.length > 0) {
        if (!this.open) this.open = { type: "thinking", thinking: "" };
        this.open.signature = entry.signature;
        this.finalized.push(this.open);
        this.open = null;
      }
    }
  }

  finalize(): ClaudeOpenAICompatSidecar["thinking_blocks"] | undefined {
    if (this.open) {
      this.finalized.push(this.open);
      this.open = null;
    }
    return this.finalized.length > 0 ? this.finalized : undefined;
  }
}

// ── Provider factory ─────────────────────────────────────────────

function createClaudeOpenAICompatProvider(opts: ProviderFactoryOpts): LLMProvider {
  const baseUrl = normalizeBaseUrl(opts.baseUrl ?? "https://api.anthropic.com");
  const url = `${baseUrl}/chat/completions`;
  const model = opts.model;
  const temperature = opts.temperature ?? 0.7;
  const readers = opts.readers;
  // Raw effort (used by adaptive-only models which accept the full effort enum
  // verbatim through output_config.effort). Budgeted effort is downgraded to
  // the {low, medium, high} subset that the explicit-budget thinking path
  // supports.
  const rawEffort: ReasoningEffort | undefined = opts.reasoningEffort;
  const budgetedEffort = rawEffort
    ? (downgradeEffort(rawEffort, ["low", "medium", "high"]) as
        | "low"
        | "medium"
        | "high")
    : undefined;
  const userMaxTokens = opts.maxTokens ?? 4096;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${opts.apiKey}`,
  };

  /**
   * Replay historical thinking_blocks (with signature) back to the API.
   * Without this, Bedrock rejects multi-turn tool_use whose preceding
   * thinking blocks are unsigned. Unsigned blocks are deliberately
   * dropped — sending a broken shape would fail the request whereas a
   * missing thinking history is recoverable (model just loses prior
   * private reasoning context).
   */
  const assistantTransform: OpenAIAssistantTransform = (apiMsg, srcMsg) => {
    const sidecar = srcMsg.providerSidecarData?.claude_openai_compat as
      | ClaudeOpenAICompatSidecar
      | undefined;
    if (!sidecar?.thinking_blocks?.length) return;
    const replay = sidecar.thinking_blocks.filter((b) => {
      if (b.type === "redacted_thinking") return typeof b.data === "string" && b.data.length > 0;
      return typeof b.signature === "string" && b.signature.length > 0;
    });
    if (replay.length > 0) {
      (apiMsg as Record<string, unknown>).thinking_blocks = replay;
    }
  };

  return {
    async shapeMessages(messages, _context) {
      return await inlineTextFiles(messages, readers);
    },

    async *chatStream(system, messages, tools, signal) {
      // `system` is stable-only; dynamic role:"system" carriers are folded
      // (two-layer smoosh) into their preceding immutable turn (see dynamic-system).
      const stable = system ?? [];

      // Build messages WITHOUT a system entry — we add a Claude-style
      // structured system below so it can carry cache_control markers.
      const openaiMessages = await messagesToOpenAI(
        foldDynamicReminders(messages),
        readers,
        undefined,
        assistantTransform,
      );

      if (stable.length > 0) {
        openaiMessages.unshift({
          role: "system",
          content: systemBlocksToOpenAIClaude(stable),
        });
      }

      annotateMessageCacheOpenAI(openaiMessages);

      const body: Record<string, unknown> = {
        model,
        messages: openaiMessages,
        stream: true,
        stream_options: { include_usage: true },
      };

      const openaiTools = toolDefsToOpenAI(tools);
      if (openaiTools) body.tools = openaiTools;

      let maxTokens = userMaxTokens;
      if (isAdaptiveOnlyModel(model)) {
        if (rawEffort) {
          body.thinking = { type: "adaptive", display: "summarized" };
          body.output_config = { effort: rawEffort };
        }
        // Adaptive-only models (Opus 4.7+) reject non-default sampling.
      } else if (budgetedEffort) {
        const budget = effortToBudget(budgetedEffort);
        body.thinking = { type: "enabled", budget_tokens: budget };
        // Bedrock requires max_tokens strictly greater than budget_tokens.
        // Caller is expected to size max_tokens to fit thinking + visible
        // output; this guard keeps the request well-formed if they forget.
        if (maxTokens <= budget) maxTokens = budget + 4096;
        body.temperature = 1;
      } else {
        body.temperature = temperature;
      }
      body.max_tokens = maxTokens;

      const res = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal,
      });
      if (!res.ok) {
        const text = await res.text();
        throwHttpApiError(res, text, "claude-openai-compat", model);
      }

      const pendingToolCalls = new Map<
        number,
        { id: string; name: string; arguments: string }
      >();
      const thinking = new ThinkingBlocksAccumulator();
      let rawUsage: Record<string, unknown> | undefined;
      let promptTokens = 0;
      let completionTokens = 0;

      try {
        for await (const { data } of parseSSE(res)) {
          if (data === "[DONE]") break;
          if (signal.aborted) break;

          let parsed: any;
          try {
            parsed = JSON.parse(data);
          } catch {
            continue;
          }

          // HTTP-200 stream that carries a mid-stream error chunk → throw (no-op otherwise).
          throwOnStreamError(parsed, "claude-openai-compat", model);

          // Usage may arrive on its own trailing chunk OR piggyback on the
          // final delta chunk (LiteLLM/forgeax does the latter). Capture
          // here but defer the actual yield to after the stream finishes so
          // the sidecar carries the complete thinking history.
          if (parsed.usage && typeof parsed.usage === "object") {
            rawUsage = parsed.usage as Record<string, unknown>;
            if (typeof rawUsage.prompt_tokens === "number") promptTokens = rawUsage.prompt_tokens;
            if (typeof rawUsage.completion_tokens === "number") completionTokens = rawUsage.completion_tokens;
          }

          const choice = parsed.choices?.[0];
          if (!choice) continue;

          const delta = choice.delta;
          if (!delta) continue;

          if (typeof delta.reasoning_content === "string" && delta.reasoning_content.length > 0) {
            yield { type: "thinking", text: delta.reasoning_content };
          }

          if (Array.isArray(delta.thinking_blocks)) {
            thinking.ingest(delta.thinking_blocks);
          }

          if (typeof delta.content === "string" && delta.content.length > 0) {
            yield { type: "text", text: delta.content };
          }

          if (Array.isArray(delta.tool_calls)) {
            for (const tc of delta.tool_calls) {
              const idx = typeof tc.index === "number" ? tc.index : 0;
              if (!pendingToolCalls.has(idx)) {
                pendingToolCalls.set(idx, {
                  id: tc.id ?? "",
                  name: tc.function?.name ?? "",
                  arguments: "",
                });
              }
              const pending = pendingToolCalls.get(idx)!;
              if (tc.id) pending.id = tc.id;
              if (tc.function?.name) pending.name = tc.function.name;
              if (tc.function?.arguments) pending.arguments += tc.function.arguments;
            }
          }

          if (
            choice.finish_reason === "tool_calls" ||
            choice.finish_reason === "stop"
          ) {
            for (const [, tc] of pendingToolCalls) {
              yield {
                type: "tool_call",
                id: tc.id,
                name: tc.name,
                arguments: tc.arguments,
              };
            }
            pendingToolCalls.clear();
          }
        }
      } finally {
        // Drain any tool calls left open at stream end (defensive — most
        // providers close with finish_reason but some proxies skip it).
        for (const [, tc] of pendingToolCalls) {
          yield {
            type: "tool_call",
            id: tc.id,
            name: tc.name,
            arguments: tc.arguments,
          };
        }
        pendingToolCalls.clear();
      }

      const finalizedThinking = thinking.finalize();
      if (rawUsage || finalizedThinking) {
        const sidecar: ClaudeOpenAICompatSidecar = {};
        if (rawUsage) sidecar.usage_raw = rawUsage;
        if (finalizedThinking) sidecar.thinking_blocks = finalizedThinking;
        yield {
          type: "provider_sidecar",
          providerSidecarData: { claude_openai_compat: sidecar },
        };
      }
      if (rawUsage) {
        yield { type: "usage", inputTokens: promptTokens, outputTokens: completionTokens };
      }
    },
  };
}

registerProvider("claude-openai-compat", createClaudeOpenAICompatProvider, {
  validateKey: requireApiKey,
});

export { createClaudeOpenAICompatProvider };
