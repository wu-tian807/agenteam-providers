// @desc Provider-side helpers for the per-loop dynamic-delta carrier.
//
// Dynamic system context (clock, todo, framework-rules, project-context, …) is
// recorded once per LLM loop as a `hook:systemPrompt` event and materialized by
// history-pipeline into a `role:"system"` carrier message that holds the FULL
// changed SystemBlock[] (both rendered text in `content` and structured form in
// `systemBlocks`). The carrier is anchored to the immutable user / tool_result
// that triggered the loop, so its bytes never change across calls → it sits in
// the cache prefix instead of poisoning the tail chunk every turn.
//
// Each provider decides how to surface those carriers:
//   - default (DeepSeek / OpenAI-compat / Gemini / Claude-compat):
//       `foldDynamicReminders` — two-layer smoosh into the preceding message.
//   - Anthropic Opus 4.8: native `{role:"system"}` (handled in anthropic.ts).
//   - GPT-5 Responses: `reconstructDynamicBlocks` → instructions, drop carriers.
import type { ContentPart, LLMMessage, SystemBlock } from "@agenteam/types";

const REMINDER_PREAMBLE =
  "**Periodic system context — NOT a user request. Continue your current task; do not pivot.**\n\n";

/** Concatenated block text of a `role:"system"` carrier message. */
function carrierText(msg: LLMMessage): string {
  return msg.content
    .filter((p): p is Extract<ContentPart, { type: "text" }> => p.type === "text")
    .map((p) => p.text)
    .join("\n\n");
}

/** Wrap carrier text as a `<system-reminder>` block for smoosh paths. */
function wrapReminder(body: string): string {
  return `<system-reminder>\n${REMINDER_PREAMBLE}${body}\n</system-reminder>`;
}

/**
 * Smoosh reminder text INTO a tool turn's own return value, not as a sibling
 * block. Merges into the last existing text part (so the tool turn keeps the
 * exact same part shape — only the result string grows); a text part is added
 * only for a media-only result. This is what keeps the fold clear of Gemini's
 * thoughtSignature pairing (signatures ride the assistant functionCall sidecar)
 * and Anthropic's tool_result integrity (one tool_result per tool_use): both key
 * off structure, never text length.
 */
function smooshIntoToolResult(msg: LLMMessage, reminder: string): LLMMessage {
  const content = msg.content.slice();
  for (let i = content.length - 1; i >= 0; i--) {
    const part = content[i];
    if (part.type === "text") {
      content[i] = { ...part, text: `${part.text}\n\n${reminder}` };
      return { ...msg, content };
    }
  }
  content.push({ type: "text", text: reminder });
  return { ...msg, content };
}

/**
 * Default two-layer smoosh: fold every `role:"system"` carrier into the message
 * immediately preceding it (the immutable turn it was anchored to):
 *   - preceding is `tool`  → smoosh into the tool_result's own value text (no
 *     extra block, so signature/integrity checks stay untouched);
 *   - preceding is `user`  → merge into that user turn (reminder ABOVE the user's
 *     own content, so the real input stays at the tail / marker position);
 *   - no foldable predecessor (conversation start or assistant tail) → emit a
 *     standalone `role:"user"` reminder so the context is never dropped.
 *
 * Returns a new list with NO `role:"system"` messages. Byte-stable because every
 * carrier is itself immutable (persisted in history).
 */
export function foldDynamicReminders(messages: LLMMessage[]): LLMMessage[] {
  const out: LLMMessage[] = [];
  for (const msg of messages) {
    if (msg.role !== "system") {
      out.push(msg);
      continue;
    }
    const reminder = wrapReminder(carrierText(msg));
    const prev = out[out.length - 1];
    if (prev && prev.role === "tool") {
      out[out.length - 1] = smooshIntoToolResult(prev, reminder);
    } else if (prev && prev.role === "user") {
      out[out.length - 1] = { ...prev, content: [{ type: "text", text: reminder }, ...prev.content] };
    } else {
      out.push({ role: "user", content: [{ type: "text", text: reminder }] });
    }
  }
  return out;
}

/**
 * Reconstruct the current full dynamic SystemBlock[] from the `role:"system"`
 * carriers in history, applying last-wins by block name (later carriers override
 * earlier ones for the same block). A retracted (tombstone) block drops its name
 * entirely — this view is rebuilt fresh each turn, so a removed slot simply
 * vanishes (no tombstone text). Output is sorted by the block's own priority so
 * the assembled order matches the original slot priority ordering. Used by
 * providers that want the full dynamic state in one place (GPT-5 Responses →
 * `instructions`).
 */
export function reconstructDynamicBlocks(messages: LLMMessage[]): SystemBlock[] {
  const byName = new Map<string, SystemBlock>();
  for (const msg of messages) {
    if (msg.role !== "system" || !msg.systemBlocks) continue;
    for (const b of msg.systemBlocks) {
      if (b.retracted) byName.delete(b.name);
      else byName.set(b.name, b);
    }
  }
  return [...byName.values()].sort((a, b) => a.priority - b.priority);
}
