/** Gemini 2.x adapter — thinkingBudget, no thought signature requirements */

import { GoogleGenAI } from "@google/genai";
import { registerProvider, requireApiKey, downgradeEffort, type ProviderFactoryOpts } from "./provider.js";
import type { LLMMessage, LLMProvider } from "./types.js";
import { annotateLLMError } from "./errors.js";
import {
  collectToolResponsesToGemini,
  toolDefsToGemini,
  contentPartsToGemini,
  describeGeminiError,
  shapeGeminiMessages,
  streamGeminiResponse,
} from "./gemini-shared.js";
import type { GeminiFileRefMap } from "./files-api-helper.js";
import type { MediaReaders } from "./media-readers.js";
import { extractTextContent } from "@agenteam/types";
import { blocksToText } from "./types.js";
import { foldDynamicReminders } from "./dynamic-system.js";

export async function messagesToGemini2(
  messages: LLMMessage[],
  readers: MediaReaders,
  fileRefMap?: GeminiFileRefMap,
): Promise<any[]> {
  const contents: any[] = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];

    if (msg.role === "tool") {
      const grouped = collectToolResponsesToGemini(messages, i);
      contents.push(grouped.content);
      i = grouped.nextIndex - 1;
      continue;
    }

    if (msg.role === "assistant") {
      const parts: any[] = [];
      const thinkingText = msg.thinking;

      if (thinkingText) {
        parts.push({ thought: true, text: thinkingText });
      }

      const textContent = extractTextContent(msg.content);
      if (textContent) {
        parts.push({ text: textContent });
      }

      if (msg.toolCalls?.length) {
        for (const tc of msg.toolCalls) {
          parts.push({ functionCall: { name: tc.name, args: tc.arguments } });
        }
      }

      if (parts.length > 0) {
        contents.push({ role: "model", parts });
      }
      continue;
    }

    contents.push({
      role: "user",
      parts: await contentPartsToGemini(msg.content, readers, fileRefMap),
    });
  }
  return contents;
}

function createGemini2Provider(opts: ProviderFactoryOpts): LLMProvider {
  const client = new GoogleGenAI({ apiKey: opts.apiKey });
  const model = opts.model;
  const temperature = opts.temperature;
  const maxOutputTokens = opts.maxTokens;
  const reasoningEffort = opts.reasoningEffort;
  const readers = opts.readers;
  // Closure-scoped: maps part srcKey → already-uploaded Files API ref.
  // Survives the lifetime of this provider instance (one-shot fallback
  // chain link); shape pipeline populates, contentPartsToGemini consumes.
  const fileRefMap: GeminiFileRefMap = new Map();

  return {
    async shapeMessages(messages, context) {
      return await shapeGeminiMessages(
        client,
        messages,
        readers,
        opts.shapeCache,
        fileRefMap,
        context,
      );
    },
    async *chatStream(system, messages, tools, signal) {
      try {
        // `system` is stable-only → systemInstruction. Dynamic blocks ride in
        // history as immutable role:"system" carriers; the default two-layer
        // smoosh folds each into its preceding (immutable) user / tool_result
        // turn so Gemini implicit context caching keeps systemInstruction + the
        // full byte-stable history cache-eligible.
        const contents = await messagesToGemini2(foldDynamicReminders(messages), readers, fileRefMap);
        const geminiTools = toolDefsToGemini(tools);

        const config: any = {
          systemInstruction: system?.length ? blocksToText(system) : undefined,
          tools: geminiTools,
          temperature,
          maxOutputTokens,
        };

        if (reasoningEffort) {
          const effective = downgradeEffort(reasoningEffort, ["low", "medium", "high"]);
          const budget = effective === "high" ? 32768 : effective === "medium" ? 16384 : 4096;
          config.thinkingConfig = {
            includeThoughts: true,
            thinkingBudget: budget,
          };
        }

        const response = await client.models.generateContentStream({ model, contents, config });

        yield* streamGeminiResponse(response, signal);
      } catch (err) {
        // Do NOT log here: this failure may be silently recovered by a
        // fallback model (e.g. auto_daily's primary→main fallback). Attach
        // structured diagnostics and rethrow; the caller decides whether to
        // log (fallback chain hooks / background tasks).
        annotateLLMError(err, { provider: "gemini", model, detail: describeGeminiError(err) });
        throw err;
      }
    },
  };
}

registerProvider("google-gemini-2", createGemini2Provider, {
  validateKey: requireApiKey,
});

export { createGemini2Provider };
