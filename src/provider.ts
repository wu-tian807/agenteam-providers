/** LLM Provider registry — config-driven adapter factory with fallback chain support */

import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { prepareMessagesForModel } from "./modality.js";
import {
  createPendingToolMessages,
  createToolResultMessage,
} from "./tool-normalizer.js";
import type { ModelSpec, ModelsConfig, ReasoningEffort, ToolSchema } from "@agenteam/types";
import { responseToAssistantMessage } from "./stream.js";
import type { LLMProvider, ShapeCache } from "./types.js";
import { defaultMediaReaders, type MediaReaders } from "./media-readers.js";
import { withRetry, calculateDelay, type RetryOptions, type RetryInfo } from "./retry.js";
import { annotateLLMError, classifyLLMError, getRecommendedDelay } from "./errors.js";
import { getSharedPaths, sleep } from "@agenteam/types";

// ── Types ────────────────────────────────────────────────────────

export interface ProviderFactoryOpts {
  model: string;
  apiKey: string;
  baseUrl?: string;
  temperature?: number;
  maxTokens?: number;
  reasoningEffort?: ReasoningEffort;
  fast?: boolean;
  /** Host-supplied byte readers for media/file ContentParts. Raw adapters
   *  capture this in closure at construction time and never reach for a
   *  package-global state. See media-readers.ts for the interface contract. */
  readers: MediaReaders;
  /** Optional cache backend for derived shape artefacts (image compression
   *  output, Gemini Files API URIs). Forwarded from `ProviderDeps.shapeCache`
   *  by `buildModelProvider` — raw adapters that compose shape helpers
   *  capture this at construction time. Undefined → helpers degrade to
   *  in-memory only. */
  shapeCache?: import("./types.js").ShapeCache;
  /** Optional escape hatch for adapters whose auth model does not fit
   *  `apiKey + baseUrl` — e.g. AWS Bedrock needs `accessKeyId / secretAccessKey
   *  / sessionToken / region`. The orchestrator stuffs whatever extra fields a
   *  given `KeyEntry` carried into this bag and the factory casts to its own
   *  shape. Other adapters ignore it. */
  extras?: Record<string, unknown>;
}

/**
 * Provider key entries — type hints for `llm_key.json` sections.
 *
 * NOTE: This union is **not** the schema source of truth — runtime validation
 * lives in `validateKeyEntry()` which dispatches to each adapter's own
 * `validateKey` registered alongside its factory (see `registerProvider`).
 *
 * The union exists purely as a TypeScript-level convenience: discriminated
 * narrowing lets test fixtures and config builders get autocomplete on the
 * right fields. Adding a new provider is **not required** to extend this
 * union — but doing so gives users typed config hints. New shapes can be
 * appended freely; nothing in `provider.ts` switches on it.
 */

interface BaseKeyEntry {
  /** Discriminant — selects the wire adapter. */
  api: KnownApiType;
  /** Model ids this section provides credentials for. */
  models: string[];
}

/**
 * Adapters that authenticate with a single bearer-style API key. Most of them
 * also support a custom endpoint via `api_base` (proxy / private deployment).
 */
interface ApiKeyAuthEntry extends BaseKeyEntry {
  api:
    | "anthropic-messages"
    | "openai-completions"
    | "openai-responses"
    | "deepseek-v4"
    | "claude-openai-compat";
  api_key: string;
  /** Override the upstream endpoint. Defaults vary per adapter (see each
   *  factory's `baseUrl ?? "..."` fallback). */
  api_base?: string;
}

/**
 * Gemini adapters (`@google/genai`) — bearer-style API key, but the SDK
 * manages endpoints internally so `api_base` is intentionally not surfaced.
 */
interface GeminiKeyEntry extends BaseKeyEntry {
  api: "google-gemini-2" | "google-gemini-3";
  api_key: string;
}

/**
 * AWS Bedrock — IAM static credentials, region-scoped. Endpoint is derived
 * from `aws_region`; there is no `api_base`. Optional `aws_session_token`
 * supports STS-issued temporary credentials (no auto-refresh — caller's job).
 */
interface BedrockKeyEntry extends BaseKeyEntry {
  api: "aws-bedrock-anthropic";
  aws_access_key_id: string;
  aws_secret_access_key: string;
  aws_region: string;
  aws_session_token?: string;
}

/**
 * Discriminated union of every supported key-section shape. TypeScript will
 * narrow access to the correct field set once you've checked `entry.api`.
 */
export type KeyEntry = ApiKeyAuthEntry | GeminiKeyEntry | BedrockKeyEntry;

/** Known `api` literal types for IDE autocomplete on hand-written configs.
 *  Runtime accepts any string that has a registered factory; this union is
 *  a hint, not a constraint. */
export type KnownApiType = KeyEntry["api"];

/**
 * Validate a key-section entry's shape.
 *
 * Pure function — no I/O, returns a list of `{field, problem}` issues so
 * callers can decide whether to throw, warn, or aggregate. Both the runtime
 * provider construction path (`buildModelProvider`, hard error on issues)
 * and the startup config script (`60-config.sh`, soft warnings) call this.
 *
 * Schema knowledge is split:
 *   - **Base checks** (here): `api` is a non-empty string, `models` is a
 *     non-empty array. Every section must satisfy these regardless of api.
 *   - **Per-provider checks** (registry): each adapter calls
 *     `registerProvider("foo", factory, { validateKey })` to declare what
 *     fields IT needs. The validator runs after base checks pass.
 *
 * Adding a new provider → write the validator in your provider file, no
 * changes here. Unknown `api` types yield a single "unknown api" issue.
 */
export function validateKeyEntry(
  entry: unknown,
): Array<{ field: string; problem: string }> {
  const issues: Array<{ field: string; problem: string }> = [];
  if (!entry || typeof entry !== "object") {
    return [{ field: "<root>", problem: "section is not an object" }];
  }
  const e = entry as Record<string, unknown>;
  if (!isNonEmptyString(e.api)) {
    issues.push({ field: "api", problem: "missing or empty" });
    return issues;
  }
  if (!Array.isArray(e.models) || e.models.length === 0) {
    issues.push({ field: "models", problem: "must be a non-empty array of model ids" });
  }

  const reg = registry.get(e.api);
  if (!reg) {
    issues.push({
      field: "api",
      problem: `unknown api type '${e.api}' — no adapter registered for this value`,
    });
    return issues;
  }
  if (reg.validateKey) {
    issues.push(...reg.validateKey(e));
  }
  return issues;
}

type ProviderFactory = (opts: ProviderFactoryOpts) => LLMProvider;

// ── Adapter Registry (keyed by api type, e.g. "google-gemini-2") ──

/**
 * Registry value bundles the factory with an optional schema validator.
 * Each adapter owns its own credential schema — `provider.ts` itself doesn't
 * know what fields any specific `api` type needs. Adding a new provider →
 * new file calls `registerProvider("foo", factory, { validateKey })`; nothing
 * in this file changes.
 */
interface RegistryEntry {
  factory: ProviderFactory;
  /** Returns issues; empty array = OK. Called by `validateKeyEntry()` after
   *  base shape checks (api / models presence) have already passed. */
  validateKey?: (entry: Record<string, unknown>) => Array<{ field: string; problem: string }>;
  /** Map a key-entry to the `extras` bag the factory expects. Default
   *  behavior (when omitted) is no extras — the adapter authenticates via
   *  the standard `apiKey` / `baseUrl` channel only. Adapters whose auth
   *  doesn't fit that mold (e.g. AWS Bedrock) override this. */
  packExtras?: (entry: Record<string, unknown>) => Record<string, unknown> | undefined;
}

const registry = new Map<string, RegistryEntry>();

export function registerProvider(
  apiType: string,
  factory: ProviderFactory,
  opts?: {
    validateKey?: RegistryEntry["validateKey"];
    packExtras?: RegistryEntry["packExtras"];
  },
): void {
  registry.set(apiType, {
    factory,
    validateKey: opts?.validateKey,
    packExtras: opts?.packExtras,
  });
}

/** Helper for adapters to write tight validators. Exported so each provider
 *  file can use the same emptiness convention. */
export function isNonEmptyString(v: unknown): v is string {
  return typeof v === "string" && v.trim() !== "";
}

/** Standard validator for adapters whose only required credential is a
 *  non-empty `api_key`. Most providers (anthropic / openai-compat / gemini /
 *  ...) just register with `{ validateKey: requireApiKey }`. */
export function requireApiKey(
  entry: Record<string, unknown>,
): Array<{ field: string; problem: string }> {
  return isNonEmptyString(entry.api_key)
    ? []
    : [{ field: "api_key", problem: "required (non-empty string)" }];
}

// ── model@keySection syntax ──────────────────────────────────────

export function parseModelSpec(raw: string): { model: string; keySection?: string } {
  const at = raw.lastIndexOf("@");
  if (at > 0 && at < raw.length - 1) {
    return { model: raw.slice(0, at), keySection: raw.slice(at + 1) };
  }
  return { model: raw };
}

// ── Key file loading ─────────────────────────────────────────────

function getKeyDir(): string {
  try {
    return getSharedPaths().keyDir();
  } catch {
    return resolve(process.cwd(), "key");
  }
}

/** Always re-read from disk so key/model changes take effect without restart. */
function loadKeyFile(): Record<string, KeyEntry> {
  const p = resolve(getKeyDir(), "llm_key.json");
  if (existsSync(p)) {
    try {
      return JSON.parse(readFileSync(p, "utf-8"));
    } catch { /* malformed JSON */ }
  }
  throw new Error("Missing key/llm_key.json — copy from llm_key.example.json");
}

// ── Model catalog loading (key/models.json) ──────────────────────

const DEFAULT_SPEC: ModelSpec = {
  input: ["text"],
  reasoning: false,
  contextWindow: 128000,
  maxOutput: 4096,
  defaultTemperature: 0.7,
};

/** Always re-read from disk so model spec changes take effect without restart. */
function loadModelCatalog(): Record<string, ModelSpec> {
  const p = resolve(getKeyDir(), "models.json");
  if (existsSync(p)) {
    try {
      return JSON.parse(readFileSync(p, "utf-8"));
    } catch { /* malformed JSON */ }
  }
  return {};
}

export function getModelSpec(model: string): ModelSpec {
  const catalog = loadModelCatalog();
  return catalog[model] ?? DEFAULT_SPEC;
}

/**
 * The usable prompt budget for a model: the full context window minus the
 * output reservation (`max_tokens`). A request must satisfy
 * `promptTokens + maxOutput <= contextWindow`, so the prompt can never exceed
 * `contextWindow - maxOutput`. This — not the raw context window — is the right
 * denominator for "how full is the context" / "when to compact" decisions;
 * using the raw window leaves a dead zone where requests already overflow
 * (prompt + maxOutput) but utilization hasn't crossed the compaction threshold.
 */
export function usableContextWindow(model: string): number {
  const spec = getModelSpec(model);
  const usable = spec.contextWindow - (spec.maxOutput ?? 0);
  return usable > 0 ? usable : spec.contextWindow;
}

// ── Effort downgrade ─────────────────────────────────────────────

/** Canonical effort ordering from lowest to highest. */
const EFFORT_ORDER: readonly ReasoningEffort[] = [
  "minimal", "low", "medium", "high", "xhigh", "max",
];

/**
 * Downgrade a reasoning effort level to the nearest supported level.
 * Searches downward first (prefer less effort), then upward if nothing below.
 */
export function downgradeEffort(
  effort: ReasoningEffort,
  supported: readonly ReasoningEffort[],
): ReasoningEffort {
  const set = new Set<ReasoningEffort>(supported);
  if (set.has(effort)) return effort;
  const idx = EFFORT_ORDER.indexOf(effort);
  for (let i = idx - 1; i >= 0; i--) {
    if (set.has(EFFORT_ORDER[i])) return EFFORT_ORDER[i];
  }
  for (let i = idx + 1; i < EFFORT_ORDER.length; i++) {
    if (set.has(EFFORT_ORDER[i])) return EFFORT_ORDER[i];
  }
  return effort;
}

// ── Resolve final model params (models.json defaults + agent.json overrides) ──

export interface ResolvedModelParams {
  /** Sampling temperature. `undefined` = model does not accept temperature (e.g. Opus 4.7+). */
  temperature?: number;
  maxTokens: number;
  reasoningEffort?: ReasoningEffort;
}

/**
 * 解析模型参数
 * @param model 模型名称
 * @param modelsConfig 模型配置
 */
export function resolveModelParams(
  model: string,
  modelsConfig?: ModelsConfig,
  spec: ModelSpec = getModelSpec(model),
): ResolvedModelParams {
  const temperature = spec.defaultTemperature !== undefined
    ? (modelsConfig?.temperature ?? spec.defaultTemperature)
    : undefined;
  const maxTokens = modelsConfig?.maxTokens ?? spec.maxOutput;
  const reasoningEffort = spec.reasoning
    ? (modelsConfig?.reasoningEffort ?? undefined)
    : undefined;

  return { temperature, maxTokens, reasoningEffort };
}

/**
 * 从 ModelsConfig 构建 RetryOptions
 */
export function buildRetryOptions(modelsConfig?: ModelsConfig): RetryOptions {
  const opts: RetryOptions = {};

  if (modelsConfig?.maxRetries !== undefined) {
    opts.maxRetries = modelsConfig.maxRetries;
  }
  if (modelsConfig?.baseDelayMs !== undefined) {
    opts.baseDelayMs = modelsConfig.baseDelayMs;
  }
  if (modelsConfig?.maxDelayMs !== undefined) {
    opts.maxDelayMs = modelsConfig.maxDelayMs;
  }

  return opts;
}

// ── Model → Section resolution (scan models arrays) ─────────────

function resolveSection(model: string): { sectionName: string; entry: KeyEntry } {
  const keys = loadKeyFile();
  for (const [name, entry] of Object.entries(keys)) {
    if (entry.models?.includes(model)) {
      return { sectionName: name, entry };
    }
  }
  throw new Error(
    `No key section contains model '${model}'. ` +
    `Add it to a section's "models" array in key/llm_key.json, ` +
    `or use "model@section" syntax to specify explicitly.`,
  );
}

// ── Dependency injection ─────────────────────────────────────────

/**
 * External dependencies a provider needs to turn a model name into a concrete
 * API call. All three default to package-shipped implementations:
 *   - `resolveKey` / `getModelSpec`: re-read `key/llm_key.json` +
 *     `key/models.json` from the shared state dir on every call.
 *   - `readers`: pure `node:fs.readFile`-backed reads (see
 *     `defaultMediaReaders`).
 *
 * Hosts override individual fields (typically `readers`, when bytes need to
 * route through a sandbox/container bridge) by passing a custom object —
 * usually built once at boot and reused across every `createProvider()` call.
 */
export interface ProviderDeps {
  /** Resolve the key section (api type, api_key, base url) for a model. */
  resolveKey(model: string, keySection?: string): { entry: KeyEntry; sectionName: string };
  /** Look up a model's capability spec (modality, context window, max output). */
  getModelSpec(model: string): ModelSpec;
  /** Read media/file bytes for ContentParts. */
  readers: MediaReaders;
  /** Optional cache backend for derived shape artefacts (image compression
   *  output, Gemini Files API URIs). When undefined, provider helpers degrade
   *  to in-memory only / re-compute every buildPrompt — host injects an
   *  `FsShapeCache` to enable cross-restart reuse. */
  shapeCache?: ShapeCache;
}

/**
 * Package-default `ProviderDeps`. File-backed key/spec lookups + node:fs
 * media readers — works out of the box for any consumer reading model inputs
 * from real host filesystem paths. Hosts with a sandbox bridge override
 * `readers` with their own implementation.
 */
export const defaultProviderDeps: ProviderDeps = {
  resolveKey(model, keySection) {
    if (keySection) {
      const entry = loadKeyFile()[keySection];
      if (!entry) {
        throw new Error(`Key section '${keySection}' not found in key/llm_key.json`);
      }
      return { entry, sectionName: keySection };
    }
    return resolveSection(model);
  },
  getModelSpec,
  readers: defaultMediaReaders,
  shapeCache: undefined,
};

// ── Public API ───────────────────────────────────────────────────

/**
 * Merge two ModelsConfig objects, with override fields taking precedence.
 * Undefined and null values in override are ignored (base value is kept).
 */
export function mergeModelsConfig(base: ModelsConfig, override: ModelsConfig): ModelsConfig {
  const result = { ...base };
  for (const [k, v] of Object.entries(override) as [keyof ModelsConfig, unknown][]) {
    if (v !== undefined && v !== null) (result as Record<string, unknown>)[k] = v;
  }
  return result;
}

export type ModelsConfigSource = ModelsConfig | (() => ModelsConfig);

export interface CreateProviderOptions {
  /** Retry policy for the fallback chain (merged with ModelsConfig fields). */
  retry?: RetryOptions;
  onRetry?: (model: string, info: RetryInfo) => void;
  onFallback?: (from: string, to: string, error: Error) => void;
  /** Inject external key / model resolution + media readers. Defaults to
   *  `defaultProviderDeps` (file-backed config + node:fs reads). */
  deps?: ProviderDeps;
}

/**
 * 创建 Provider —— 唯一入口。Provider 自带模型链 + 重试 + 降级能力。
 *
 * - 传静态 `ModelsConfig`：单模型直接包装，多模型自动组成模型链。
 * - 传 `() => ModelsConfig`：每次 chatStream 重新读取配置（模型列表 / 重试 /
 *   temperature 等），因此 agent.json 改动无需重建 provider；配合 onRetry /
 *   onFallback 钩子用于长生命周期 agent。
 * - `options.deps`：可选。省略则用 `defaultProviderDeps`（key 文件解析 +
 *   node:fs media reads）。需要 sandbox 桥的 host 传入自定义 deps。
 */
export function createProvider(
  source: ModelsConfigSource,
  options?: CreateProviderOptions,
): LLMProvider {
  const deps = options?.deps ?? defaultProviderDeps;

  if (typeof source === "function" || options?.onRetry || options?.onFallback) {
    const getMc = typeof source === "function" ? source : () => source;
    return buildChainProvider(getMc, options, deps);
  }

  const modelsConfig = source;
  const raw = Array.isArray(modelsConfig.model) ? modelsConfig.model : [modelsConfig.model ?? ""];
  const valid = raw.filter(Boolean);
  if (valid.length > 1) return buildChainProvider(() => modelsConfig, options, deps);
  if (!valid[0]) throw new Error("No model specified in ModelsConfig");

  return buildModelProvider(modelsConfig, valid[0], deps);
}

/** Wrap a single resolved model's adapter (one link in the model chain). */
function buildModelProvider(
  modelsConfig: ModelsConfig,
  modelSpecRaw: string,
  deps: ProviderDeps,
): LLMProvider {
  const { model, keySection } = parseModelSpec(modelSpecRaw);
  const { entry, sectionName } = deps.resolveKey(model, keySection);

  const apiType = entry.api;
  const reg = registry.get(apiType);
  if (!reg) {
    throw new Error(
      `No adapter registered for api type '${apiType}'. ` +
      `Available: ${[...registry.keys()].join(", ")}`,
    );
  }
  const factory = reg.factory;

  const issues = validateKeyEntry(entry);
  if (issues.length) {
    const detail = issues.map((i) => `${i.field}: ${i.problem}`).join("; ");
    throw new Error(
      `Invalid llm_key.json section '${sectionName}' (model '${model}'): ${detail}`,
    );
  }

  // Auth fields: `apiKey` + `baseUrl` are the common channel; adapters with
  // exotic auth (AWS Bedrock IAM, etc.) opt into `packExtras` to surface
  // their own credential bag. The conditional read of `api_key` / `api_base`
  // is kept here only because they're cross-cutting standard names —
  // anything else lives in `extras`.
  const e = entry as unknown as Record<string, unknown>;
  const apiKey = typeof e.api_key === "string" ? e.api_key : "";
  const baseUrl = typeof e.api_base === "string" ? e.api_base : undefined;
  const extras = reg.packExtras ? reg.packExtras(e) : undefined;

  const resolvedSpec = deps.getModelSpec(model);
  const params = resolveModelParams(model, modelsConfig, resolvedSpec);
  const provider = factory({
    model,
    apiKey,
    baseUrl,
    temperature: params.temperature,
    maxTokens: params.maxTokens,
    reasoningEffort: params.reasoningEffort,
    fast: modelsConfig.fast,
    readers: deps.readers,
    shapeCache: deps.shapeCache,
    extras,
  });
  return {
    ...provider,
    // Wrapper transparently delegates to provider hooks. Media hygiene is
    // handled at the storage layer (event-blob.ts size-based externalize on
    // write, media-storage.ts magic-byte mime sniff on read) — not here.
    async shapeMessages(messages, context) {
      return provider.shapeMessages
        ? await provider.shapeMessages(messages, context)
        : messages;
    },
    materializeAssistantMessage(response, options) {
      return provider.materializeAssistantMessage
        ? provider.materializeAssistantMessage(response, options)
        : responseToAssistantMessage(response, options);
    },
    materializePendingToolMessages(toolCalls, options) {
      return provider.materializePendingToolMessages
        ? provider.materializePendingToolMessages(toolCalls, options)
        : createPendingToolMessages(toolCalls).map((msg) => ({ ...msg, ts: options.ts }));
    },
    materializeToolResult(toolCall, result, options) {
      return provider.materializeToolResult
        ? provider.materializeToolResult(toolCall, result, options)
        : { ...createToolResultMessage(toolCall, result), ts: options.ts };
    },
    async *chatStream(system, messages, tools, signal) {
      const filtered = tools.filter(t => !t.modelFilter || t.modelFilter(model));
      try {
        for await (const chunk of provider.chatStream(
          system,
          prepareMessagesForModel(messages, resolvedSpec.input),
          filtered,
          signal,
        )) {
          yield chunk.type === "usage" ? { ...chunk, model } : chunk;
        }
      } catch (err) {
        annotateLLMError(err, { provider: apiType, model, keySection: sectionName });
        throw err;
      }
    },
  };
}

/**
 * 构建模型链 provider —— provider 内建的重试 + 降级能力（内部实现，对外统一走
 * `createProvider`）。
 *
 * 每次 chatStream 调用时从 getModelsConfig() 读取最新配置（模型列表、重试策略、
 * temperature 等），因此 agent.json 的修改无需重建 provider 即可生效。
 *
 * 降级优先策略：每轮依次尝试链上所有模型（各试一次）；
 * 一轮全部失败后等待（指数退避），再进行下一轮。
 * 流开始后（已 yield chunk）不再降级，直接抛出。
 */
function buildChainProvider(
  getModelsConfig: () => ModelsConfig,
  options: CreateProviderOptions | undefined,
  deps: ProviderDeps,
): LLMProvider {
  const { onRetry, onFallback } = options ?? {};

  const resolveModels = () => {
    const mc = getModelsConfig();
    const raw = Array.isArray(mc.model) ? mc.model : [mc.model ?? ""];
    const models = raw.filter((m) => {
      if (!m) return false;
      try {
        const { model, keySection } = parseModelSpec(m);
        deps.resolveKey(model, keySection);
        return true;
      } catch {
        console.warn(`model "${m}" not found in llm_key.json — skipped from fallback chain`);
        return false;
      }
    });
    if (models.length === 0 && raw.some(Boolean)) {
      console.error(`all models in fallback chain are invalid: [${raw.join(", ")}]`);
    }
    return { mc, models };
  };

  const getFirstModelProvider = () => {
    const { mc, models } = resolveModels();
    const firstModel = models.find(Boolean);
    return firstModel ? buildModelProvider({ ...mc, model: firstModel }, firstModel, deps) : null;
  };

  return {
    async shapeMessages(messages, context) {
      // shapeMessages runs once per buildPrompt against the active (first)
      // model. Composing every chain member's hook would over-shape (e.g.
      // Gemini Files API upload then Anthropic image cache on the same
      // bytes). Chain fallback at chatStream time accepts already-shaped
      // messages — same trade-off as the previous prepareInboundMessages
      // path, but now without writing any of it back to the WAL.
      const provider = getFirstModelProvider();
      return provider?.shapeMessages
        ? await provider.shapeMessages(messages, context)
        : messages;
    },
    materializeAssistantMessage(response, options) {
      const provider = getFirstModelProvider();
      return provider
        ? provider.materializeAssistantMessage!(response, options)
        : responseToAssistantMessage(response, options);
    },
    materializePendingToolMessages(toolCalls, options) {
      const provider = getFirstModelProvider();
      return provider
        ? provider.materializePendingToolMessages!(toolCalls, options)
        : createPendingToolMessages(toolCalls).map((msg) => ({ ...msg, ts: options.ts }));
    },
    materializeToolResult(toolCall, result, options) {
      const provider = getFirstModelProvider();
      return provider
        ? provider.materializeToolResult!(toolCall, result, options)
        : { ...createToolResultMessage(toolCall, result), ts: options.ts };
    },
    async *chatStream(system, messages, tools, signal) {
      const { mc, models } = resolveModels();
      if (!models[0]) throw new Error("No model specified in ModelsConfig");

      const retryOpts: RetryOptions = { ...options?.retry, ...buildRetryOptions(mc) };
      const maxRounds = (retryOpts.maxRetries ?? 5) + 1;
      let lastError: Error | undefined;
      let round = 0, hasRoundError = false;

      for (let i = 0; i < models.length; i++) {
        const modelName = models[i];
        let chunksYielded = false;

        try {
          const provider = buildModelProvider({ ...mc, model: modelName }, modelName, deps);
          const stream = await withRetry(
            async () => {
              const it = provider.chatStream(system, messages, tools, signal)[Symbol.asyncIterator]();
              return { it, first: await it.next() };
            },
            { maxRetries: 0, signal },
          );

          if (!stream.first.done) { chunksYielded = true; yield stream.first.value; }

          for (let r = await stream.it.next(); !r.done; r = await stream.it.next()) {
            yield r.value;
          }
          return;
        } catch (err: unknown) {
          const classified = classifyLLMError(err);
          if (signal.aborted || classified.kind === "aborted") throw classified.error;
          if (chunksYielded) throw classified.error;
          lastError = classified.error;
          hasRoundError = true;
          if (i < models.length - 1) {
            onFallback?.(modelName, models[i + 1], classified.error);
          } else if (hasRoundError && ++round < maxRounds) {
            const delayMs = calculateDelay(round - 1, retryOpts.baseDelayMs ?? 1000, retryOpts.maxDelayMs ?? 30000, true, getRecommendedDelay(lastError));
            onRetry?.(modelName, { attempt: round, maxRetries: maxRounds - 1, error: lastError!, delayMs, willRetry: true });
            await sleep(delayMs, signal);
            hasRoundError = false;
            i = -1;
          }
        }
      }

      throw lastError ?? new Error("All models in fallback chain failed");
    },
  };
}
