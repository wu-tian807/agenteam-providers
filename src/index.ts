// @agenteam/providers — host-agnostic LLM provider abstractions.
//
// This is the flat public surface. Anything not re-exported here is considered
// internal to the package.

export * from "./errors.js";
export * from "./retry.js";
export * from "./types.js";
export * from "./dynamic-system.js";
export * from "./stream.js";
export * from "./media-readers.js";
export * from "./modality.js";
export * from "./tool-normalizer.js";
export * from "./media-mime-sniff.js";
export * from "./image-compression.js";
export * from "./provider.js";

// Adapter helpers — exposed for white-box tests / advanced consumers that
// want to hand-roll a request payload without going through `createProvider`.
// `create*Provider` factories themselves stay accessible too.
export {
  contentToAnthropic,
  messagesToAnthropic,
  createAnthropicProvider,
} from "./anthropic.js";
export {
  toolDefsToOpenAI,
  contentToOpenAI,
  messagesToOpenAI,
  openAICompatStream,
  normalizeBaseUrl,
  createOpenAICompatProvider,
  type OpenAIAssistantTransform,
  type OpenAIStreamOpts,
} from "./openai-compat.js";
export {
  messagesToResponseInput,
  openAIResponseStream,
  createOpenAIResponseProvider,
} from "./openai-response.js";
export {
  messagesToGemini2,
  createGemini2Provider,
} from "./gemini2.js";

// Side-effect: register all built-in adapters with the registry. Importing
// `@agenteam/providers` from anywhere in the host suffices to make
// `createProvider` work for every wired-in API type. Done last so the registry
// + factory types above are fully exported before adapter modules import them.
import "./register-all.js";
