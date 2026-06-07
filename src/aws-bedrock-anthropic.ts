/**
 * AWS Bedrock — Anthropic Messages adapter.
 *
 * Wraps the same wire-format that `anthropic.ts` produces (Messages API body
 * shape, including thinking / cache_control / fast mode), and ships it through
 * Bedrock's `invoke-with-response-stream` endpoint instead of
 * `api.anthropic.com/v1/messages`. The two changes are auth (SigV4 Authorization
 * header instead of `x-api-key`) and stream framing (AWS event-stream binary
 * frames instead of plain text/event-stream). Body construction and event
 * consumption are reused verbatim — `consumeAnthropicEvents` is the shared
 * post-parse loop.
 *
 * Scope (see `docs/changelog/2026-06-07-aws-bedrock-provider.md`):
 *  - Static IAM credentials in `llm_key.json` only — no role assume / SSO /
 *    IMDS / `~/.aws/credentials`.
 *  - Anthropic-format body only (`invoke-with-response-stream`). The Bedrock
 *    Converse API is intentionally not implemented yet.
 *  - No streaming-payload chunk signing (Bedrock invoke-with-response-stream
 *    is not a streaming-PUT — entire request body is signed once up front).
 */

import { registerProvider, isNonEmptyString, type ProviderFactoryOpts } from "./provider.js";
import type { LLMProvider } from "./types.js";
import {
  messagesToAnthropic,
  toolDefsToAnthropic,
  systemBlocksToAnthropic,
  annotateMessageCache,
  consumeAnthropicEvents,
  isAdaptiveOnlyModel,
  supportsMidConvoSystem,
  supportsFastMode,
} from "./anthropic.js";
import { downgradeEffort } from "./provider.js";
import { sigv4Sign } from "./aws-sigv4.js";
import { parseAwsEventStream } from "./aws-eventstream.js";
import { foldDynamicReminders } from "./dynamic-system.js";
import { throwHttpApiError } from "./errors.js";

interface BedrockExtras {
  awsAccessKeyId?: string;
  awsSecretAccessKey?: string;
  awsSessionToken?: string;
  awsRegion?: string;
}

function buildBedrockUrl(region: string, modelId: string): string {
  // encodeURIComponent handles dots / slashes that some inference profile ids
  // contain (e.g. "us.anthropic.claude-sonnet-4-6"). Bedrock accepts both
  // encoded and unencoded forms in the path; encoding is the safe default.
  return (
    `https://bedrock-runtime.${region}.amazonaws.com` +
    `/model/${encodeURIComponent(modelId)}/invoke-with-response-stream`
  );
}

function createBedrockAnthropicProvider(opts: ProviderFactoryOpts): LLMProvider {
  const { readers } = opts;
  const extras = (opts.extras ?? {}) as BedrockExtras;
  const accessKeyId = extras.awsAccessKeyId;
  const secretAccessKey = extras.awsSecretAccessKey;
  const region = extras.awsRegion;

  if (!accessKeyId || !secretAccessKey || !region) {
    throw new Error(
      "aws-bedrock-anthropic: missing required credentials. Set " +
      "aws_access_key_id, aws_secret_access_key, and aws_region in llm_key.json.",
    );
  }

  const modelId = opts.model;
  const url = buildBedrockUrl(region, modelId);
  const temperature = opts.temperature ?? 0.7;
  const maxTokens = opts.maxTokens ?? 4096;
  const reasoningEffort = opts.reasoningEffort;
  const fast = opts.fast;

  return {
    async prepareInboundMessages(messages, _context) {
      return messages;
    },
    async *chatStream(system, messages, tools, signal) {
      const stable = system ?? [];
      const useNative = supportsMidConvoSystem(modelId);

      const anthropicMessages = await messagesToAnthropic(
        useNative ? messages : foldDynamicReminders(messages),
        readers,
      );
      const anthropicTools = toolDefsToAnthropic(tools);

      // Bedrock-specific: model id rides in URL, body uses `anthropic_version`
      // marker required by Bedrock's Anthropic backend (NOT the public Anthropic
      // API's "anthropic-version" HTTP header). Everything else mirrors
      // anthropic.ts:createAnthropicProvider's body construction.
      const body: any = {
        anthropic_version: "bedrock-2023-05-31",
        messages: anthropicMessages,
        max_tokens: maxTokens,
      };
      if (stable.length) {
        body.system = systemBlocksToAnthropic(stable);
      }
      annotateMessageCache(anthropicMessages);
      if (anthropicTools) body.tools = anthropicTools;

      if (isAdaptiveOnlyModel(modelId)) {
        if (reasoningEffort) {
          body.thinking = { type: "adaptive", display: "summarized" };
          body.output_config = { effort: reasoningEffort };
        }
      } else {
        if (reasoningEffort) {
          const effective = downgradeEffort(reasoningEffort, ["low", "medium", "high"]);
          const budget = effective === "high" ? 32768 : effective === "medium" ? 16384 : 8192;
          body.thinking = { type: "enabled", budget_tokens: budget };
          if ((body.max_tokens as number) <= budget) {
            body.max_tokens = budget + 4096;
          }
          body.temperature = 1;
        } else {
          body.temperature = temperature;
        }
      }
      if (fast && supportsFastMode(modelId)) {
        // NOTE: Bedrock may or may not honour Anthropic's fast-mode beta header
        // / body.speed depending on the model version provisioned in your AWS
        // account. We pass it through; if rejected, drop in your config.
        body.speed = "fast";
      }

      const bodyStr = JSON.stringify(body);
      const signedHeaders = sigv4Sign(
        { accessKeyId, secretAccessKey, sessionToken: extras.awsSessionToken },
        {
          method: "POST",
          url,
          region,
          service: "bedrock",
          body: bodyStr,
          headers: {
            "content-type": "application/json",
            // Tell Bedrock we accept the AWS event-stream binary framing.
            accept: "application/vnd.amazon.eventstream",
          },
        },
      );

      const res = await fetch(url, {
        method: "POST",
        headers: signedHeaders,
        body: bodyStr,
        signal,
      });
      if (!res.ok) {
        const text = await res.text();
        throwHttpApiError(res, text, "aws-bedrock-anthropic", modelId);
      }

      // Bedrock event-stream → already-parsed `{event, data}` chunks.
      // BUT: AWS wraps every Anthropic event under a single AWS event-name of
      // `chunk`, with the real Anthropic event type living in the inner JSON's
      // `type` field. consumeAnthropicEvents falls back to `parsed.type` only
      // when `event` is nullish, so we drop Bedrock's outer `chunk` label here
      // to let the inner `type` drive the switch.
      async function* unwrapped() {
        for await (const { event, data } of parseAwsEventStream(res)) {
          yield { event: event === "chunk" ? undefined : event, data };
        }
      }
      yield* consumeAnthropicEvents(
        unwrapped(),
        signal,
        "aws-bedrock-anthropic",
        modelId,
      );
    },
  };
}

registerProvider("aws-bedrock-anthropic", createBedrockAnthropicProvider, {
  /** Bedrock authenticates via IAM static credentials, not `api_key`. The
   *  region is part of the wire endpoint (URL is region-derived) so it must
   *  also be present at config time. `aws_session_token` is optional —
   *  required only for STS-issued temporary credentials. */
  validateKey: (entry) => {
    const issues: Array<{ field: string; problem: string }> = [];
    if (!isNonEmptyString(entry.aws_access_key_id)) {
      issues.push({ field: "aws_access_key_id", problem: "required for AWS Bedrock" });
    }
    if (!isNonEmptyString(entry.aws_secret_access_key)) {
      issues.push({ field: "aws_secret_access_key", problem: "required for AWS Bedrock" });
    }
    if (!isNonEmptyString(entry.aws_region)) {
      issues.push({ field: "aws_region", problem: "required for AWS Bedrock" });
    }
    return issues;
  },
  /** Hand the IAM credentials to the factory through the `extras` channel —
   *  the standard `apiKey` slot stays empty for this provider. */
  packExtras: (entry) => ({
    awsAccessKeyId: entry.aws_access_key_id,
    awsSecretAccessKey: entry.aws_secret_access_key,
    awsSessionToken: entry.aws_session_token,
    awsRegion: entry.aws_region,
  }),
});

export { createBedrockAnthropicProvider };
