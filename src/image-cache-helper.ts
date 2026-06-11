// @desc Shared shape helper: cache-aware image compression for image / image_file parts.

import type { ContentPart, LLMMessage } from "@agenteam/types";
import type { MediaReaders } from "./media-readers.js";
import type { ShapeCache } from "./types.js";
import {
  exceedsImageLimit,
  exceedsImageDimensions,
  fitImageToPolicy,
  type ImagePreflightPolicy,
} from "./image-compression.js";
import { computeSourceKey, computePolicyHash } from "./shape-cache.js";

const MIME_EXT: Record<string, string> = {
  "image/jpeg": "jpg",
  "image/png": "png",
  "image/webp": "webp",
  "image/gif": "gif",
};

function extOf(mime: string): string {
  return MIME_EXT[mime] ?? mime.split("/")[1] ?? "bin";
}

/**
 * Replace each oversized image / image_file part with a cache-resolved
 * compressed variant. Pure shape — wire format conversion still happens
 * inside the provider's `convertPartTo*`. Output part types are preserved
 * (inline → inline; path-based → path-based with cached path).
 *
 * Behaviour matrix:
 *   - Part within `policy` limits → unchanged (no read, no cache hit, no
 *     compression). This keeps the helper cheap on the common path.
 *   - Part exceeds limits, `shapeCache` provided:
 *       compute srcKey + policyHash → lookupImage. Hit: rewrite bytes
 *       from cache. Miss: read bytes (if path-based), fitImageToPolicy,
 *       writeImage, rewrite bytes.
 *   - Part exceeds limits, `shapeCache` undefined: degrade to direct
 *     fitImageToPolicy on every call (functionally equivalent to the
 *     pre-cache wire-time compression — no regression, just no reuse).
 *   - fitImageToPolicy returns null (cannot fit): part left unchanged
 *     so the downstream `convertPartTo*` emits its own `[image omitted: ...]`
 *     placeholder against the original bytes.
 */
export async function applyImageCache(
  messages: LLMMessage[],
  readers: MediaReaders,
  shapeCache: ShapeCache | undefined,
  policy: ImagePreflightPolicy,
  providerName: string,
  signal: AbortSignal,
): Promise<LLMMessage[]> {
  const policyHash = computePolicyHash(policy);
  let touched = false;
  const out = await Promise.all(messages.map(async (msg) => {
    if (!Array.isArray(msg.content)) return msg;
    let msgTouched = false;
    const parts: ContentPart[] = [];
    for (const part of msg.content) {
      if (signal.aborted) {
        parts.push(part);
        continue;
      }
      if (part.type !== "image" && part.type !== "image_file") {
        parts.push(part);
        continue;
      }
      const replaced = await maybeReplace(
        part,
        readers,
        shapeCache,
        policy,
        policyHash,
        providerName,
      );
      if (replaced && replaced !== part) {
        msgTouched = true;
        parts.push(replaced);
      } else {
        parts.push(part);
      }
    }
    if (!msgTouched) return msg;
    touched = true;
    return { ...msg, content: parts };
  }));
  return touched ? out : messages;
}

async function maybeReplace(
  part: Extract<ContentPart, { type: "image" | "image_file" }>,
  readers: MediaReaders,
  shapeCache: ShapeCache | undefined,
  policy: ImagePreflightPolicy,
  policyHash: string,
  providerName: string,
): Promise<ContentPart | null> {
  const srcKey = computeSourceKey(part);

  // Read bytes — for image_file via host reader, for inline via base64
  // decode. We need bytes to (a) probe size, (b) feed compression on miss.
  let bytes: Buffer;
  let mimeType: string;
  try {
    const r = await readers.readMediaBytes(part);
    bytes = r.bytes;
    mimeType = r.mimeType;
  } catch {
    return null; // unreadable → leave untouched, downstream emits placeholder
  }

  // Cheap path: already within policy. No cache work.
  if (
    !exceedsImageLimit(bytes, policy) &&
    !(await exceedsImageDimensions(bytes, policy))
  ) {
    return null;
  }

  // Cache lookup — provisional ext from current mimeType. The actual
  // cache filename uses the compression OUTPUT ext (recorded at write
  // time below), so a `lookupImage(ext=jpg)` after a previous webp write
  // would miss. To keep the lookup robust across same-policy calls, we
  // probe a small set of candidate exts. Cheap because most images have
  // one or two plausible target formats.
  if (shapeCache) {
    for (const candidateExt of candidateOutputExts(mimeType)) {
      try {
        const hit = await shapeCache.lookupImage({
          providerName,
          srcKey,
          policyHash,
          ext: candidateExt,
        });
        if (hit) {
          return rewritePart(part, hit.bytes, extToMime(candidateExt), hit.path);
        }
      } catch { /* swallow — treat as miss */ }
    }
  }

  // Miss → compress.
  if (!policy.compressOversized) return null;
  const fit = await fitImageToPolicy(bytes, mimeType, policy);
  if (!fit || exceedsImageLimit(fit.bytes, policy)) return null;

  const ext = extOf(fit.mimeType);
  let cachedPath: string | undefined;
  if (shapeCache) {
    try {
      const w = await shapeCache.writeImage({
        providerName,
        srcKey,
        policyHash,
        ext,
        bytes: fit.bytes,
      });
      cachedPath = w.path;
    } catch { /* cache write failure is non-fatal */ }
  }

  return rewritePart(part, fit.bytes, fit.mimeType, cachedPath);
}

function rewritePart(
  part: Extract<ContentPart, { type: "image" | "image_file" }>,
  bytes: Buffer,
  mimeType: string,
  cachedPath?: string,
): ContentPart {
  if (part.type === "image") {
    return { type: "image", data: bytes.toString("base64"), mimeType };
  }
  // image_file — keep type but swap path to cached file when available.
  if (cachedPath) {
    return { ...part, path: cachedPath, mimeType };
  }
  // No cache, but compression succeeded — degrade to inline image so
  // downstream uses the compressed bytes (otherwise the path still
  // points at the oversized original).
  return { type: "image", data: bytes.toString("base64"), mimeType };
}

function candidateOutputExts(inputMime: string): string[] {
  const ext = extOf(inputMime);
  const exts = new Set<string>([ext, "webp", "jpg", "png"]);
  return [...exts];
}

function extToMime(ext: string): string {
  for (const [mime, e] of Object.entries(MIME_EXT)) {
    if (e === ext) return mime;
  }
  return `image/${ext}`;
}
