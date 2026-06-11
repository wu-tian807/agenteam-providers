// @desc Pure helpers (no I/O) for ShapeCache key derivation: source key + policy hash.

import { createHash } from "node:crypto";
import type { ContentPart } from "@agenteam/types";
import type { ImagePreflightPolicy } from "./image-compression.js";

/**
 * Derive a stable 16-char source key for a media-bearing ContentPart.
 *
 * Two regimes:
 *   - **Inline** (`{ type: "image" | "audio" | "video", data: <base64> }`):
 *     hash the decoded bytes — content-addressed end-to-end.
 *   - **Path-based** (`{ type: "*_file", path }`):
 *     hash the path string. Production paths come from
 *     `medias/{sha256-16}.bin` storage which is itself content-addressed,
 *     so a path change implies a content change. Inline `path` reuse with
 *     different bytes (e.g. `/tmp/foo.png`) would collide; that's why the
 *     storage layer normalises to content-addressed paths before this hook
 *     ever sees the part.
 *
 * No fs I/O — `path`-based variants do NOT stat the file. This keeps the
 * helper synchronous-on-bytes and lets callers compute the key BEFORE
 * deciding whether to read bytes (e.g. files-api-helper checks the cache
 * first and only reads bytes on miss).
 */
export function computeSourceKey(part: ContentPart): string {
  const hash = createHash("sha256");
  if ("path" in part && typeof part.path === "string") {
    hash.update("path:");
    hash.update(part.path);
    return hash.digest("hex").slice(0, 16);
  }
  if ("data" in part && typeof part.data === "string") {
    hash.update("inline:");
    hash.update(Buffer.from(part.data, "base64"));
    return hash.digest("hex").slice(0, 16);
  }
  // Defensive — unknown shape, fall back to a JSON-stable digest so callers
  // still get a deterministic key (cache will simply re-derive next call).
  hash.update("part:");
  hash.update(JSON.stringify(part));
  return hash.digest("hex").slice(0, 16);
}

/**
 * Stable 8-char policy fingerprint. Any visible field of `ImagePreflightPolicy`
 * that influences output (size caps, dimension cap, mime-type whitelist,
 * compress flag) feeds the digest. `supportedMimeTypes` is a Set — sort its
 * members before hashing to keep ordering-independent.
 *
 * Use case: policy-versioned cache filenames. Bump a cap → policyHash flips
 * → previous cached artefacts naturally fall out without explicit invalidation.
 */
export function computePolicyHash(policy: ImagePreflightPolicy): string {
  const stable = {
    maxBytes: policy.maxBytes ?? null,
    maxBase64Bytes: policy.maxBase64Bytes ?? null,
    maxLongEdge: policy.maxLongEdge ?? null,
    compressOversized: policy.compressOversized ?? null,
    supportedMimeTypes: [...policy.supportedMimeTypes].sort(),
  };
  return createHash("sha256").update(JSON.stringify(stable)).digest("hex").slice(0, 8);
}
