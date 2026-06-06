# @agenteam/providers

Raw LLM provider abstractions and SDK adapters for the AgenTeam framework.

This package is the **host-agnostic core** of the LLM stack: provider
interface (`LLMProvider`), wire-shape types (`LLMMessage`/`StreamEvent`/
`ToolSchema`), error classification + retry plumbing, dynamic-system-block
helpers, and the streaming response materializer.

It is intentionally narrow — it knows nothing about the host filesystem,
sandbox, capability registry, or pack runtime. Anything that needs `fs`, the
sandbox bridge, or per-instance state lives in the host (e.g. the main
`agenteam_os` repo) and is composed on top.

## Layout

```
src/
  errors.ts          – TerminalLLMError / RetryableLLMError + classifier + HTTP helpers
  retry.ts           – withRetry wrapper (exponential backoff + jitter + Retry-After)
  types.ts           – LLMProvider / LLMResponse / StreamEvent / ResponseKind
  dynamic-system.ts  – foldDynamicReminders / reconstructDynamicBlocks
  stream.ts          – SSE parser + responseToAssistantMessage materializer
  internal/
    content.ts       – minimal ContentPart normalizer (kept here to avoid host
                       deps; will be deduped against @agenteam/types later)
  index.ts           – flat public surface
```

Wire-shape primitives (`LLMMessage`, `SystemBlock`, `LLMToolCall`,
`ProviderSidecarData`, `ContentPart`, `ToolSchema`, thinking helpers) live
upstream in `@agenteam/types` so channels (e.g. `ink-renderer`) can talk the
same shape without depending on this package.

## Stability

This package is **private / pre-1.0**. The public surface will keep moving as
host-coupled providers (anthropic / gemini / openai-compat / etc.) and the
provider registry get pulled out of the host repo.

## License

Apache-2.0
