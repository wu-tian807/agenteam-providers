// @desc Shape helper: Gemini Files API uploads with fs-cache + closure-scoped in-memory map.

import { GoogleGenAI, type File as GoogleFile } from "@google/genai";
import type { ContentPart, LLMMessage } from "@agenteam/types";
import { isFileMediaContentPart, isInlineMediaContentPart } from "@agenteam/types";
import type { MediaReaders } from "./media-readers.js";
import type { ShapeCache, CachedFilesApiEntry } from "./types.js";
import { computeSourceKey } from "./shape-cache.js";

/** Files API upload threshold. Inline data <= 20MB stays inline; > 20MB
 *  goes through the Files API which keeps the request body small. */
export const GEMINI_INLINE_FILE_API_THRESHOLD_BYTES = 20 * 1024 * 1024;

/** Lookup safety margin — refuse cache entries within this many ms of expiry. */
const FILES_API_FRESHNESS_MARGIN_MS = 60 * 60 * 1000; // 1h

/** Conservative fallback TTL when SDK doesn't return expirationTime.
 *  Google's documented default is ~48h; we shave 1h so the next-turn
 *  freshness check (with the same margin) doesn't flap. */
const FILES_API_FALLBACK_TTL_MS = 47 * 60 * 60 * 1000;

/**
 * In-memory ref a Gemini provider keeps in closure to map already-uploaded
 * source bytes to their Files API URI. `contentPartsToGemini` consults this
 * at wire-format time via the part's srcKey.
 */
export interface GoogleFileRef {
  uri: string;
  mimeType: string;
  name?: string;
  /** Absolute timestamp ms — used for in-session freshness re-check. */
  expiresAt?: number;
}

export type GeminiFileRefMap = Map<string, GoogleFileRef>;

/**
 * Walk every message; for each `> 20MB` media/file part:
 *   1. compute srcKey
 *   2. if `fileRefMap` already has a fresh entry → skip
 *   3. else `shapeCache.lookupFilesApi` (1h freshness margin) → write
 *      result into `fileRefMap`, skip
 *   4. else read bytes, upload via `client.files.upload`, persist via
 *      `shapeCache.writeFilesApi`, write `fileRefMap`
 *
 * Messages are never mutated — the fileRef binding is implicit through
 * `fileRefMap[srcKey]`, which `contentPartsToGemini` resolves when
 * serialising parts. Upload failure is non-fatal: the part falls through
 * to the inline path (which will then likely emit `[unsupported]` against
 * Gemini's request size cap, the same failure mode as before this helper
 * existed, just without the WAL pollution of the old prepareInbound path).
 */
export async function applyGeminiFilesApi(
  messages: LLMMessage[],
  readers: MediaReaders,
  shapeCache: ShapeCache | undefined,
  fileRefMap: GeminiFileRefMap,
  client: GoogleGenAI,
  signal: AbortSignal,
): Promise<LLMMessage[]> {
  for (const msg of messages) {
    if (signal.aborted) break;
    if (!Array.isArray(msg.content)) continue;
    for (const part of msg.content) {
      if (signal.aborted) break;
      await maybeUploadPart(part, readers, shapeCache, fileRefMap, client);
    }
  }
  return messages; // shape is purely side-effecting on fileRefMap
}

async function maybeUploadPart(
  part: ContentPart,
  readers: MediaReaders,
  shapeCache: ShapeCache | undefined,
  fileRefMap: GeminiFileRefMap,
  client: GoogleGenAI,
): Promise<void> {
  // Only media + file-bearing parts are upload candidates.
  const isMedia = isFileMediaContentPart(part) || isInlineMediaContentPart(part);
  const isFile = part.type === "text_file" || part.type === "file";
  if (!isMedia && !isFile) return;

  const srcKey = computeSourceKey(part);

  // Fast path: already in memory and still fresh.
  const cached = fileRefMap.get(srcKey);
  if (cached && isFresh(cached.expiresAt)) return;
  if (cached) fileRefMap.delete(srcKey); // expired in-session — re-derive

  // FS cache lookup before reading bytes (saves a read on hit).
  if (shapeCache) {
    try {
      const hit = await shapeCache.lookupFilesApi({ srcKey });
      if (hit) {
        fileRefMap.set(srcKey, {
          uri: hit.uri,
          mimeType: hit.mimeType,
          name: hit.name,
          expiresAt: hit.expiresAt,
        });
        return;
      }
    } catch { /* miss */ }
  }

  // Need bytes — read + size-gate.
  let bytes: Buffer;
  let mimeType: string;
  try {
    if (isMedia) ({ bytes, mimeType } = await readers.readMediaBytes(part as any));
    else ({ bytes, mimeType } = await readers.readFileBytes(part as any));
  } catch {
    return; // unreadable — let downstream emit placeholder
  }
  if (bytes.byteLength <= GEMINI_INLINE_FILE_API_THRESHOLD_BYTES) return;

  // Upload.
  let uploaded: GoogleFile;
  try {
    uploaded = await client.files.upload({
      file: new globalThis.Blob([new Uint8Array(bytes)], { type: mimeType }),
      config: { mimeType },
    });
  } catch {
    return; // transient — retry next buildPrompt
  }
  if (!uploaded.uri) return;

  const expiresAt = deriveExpiresAt(uploaded);
  const ref: GoogleFileRef = {
    uri: uploaded.uri,
    mimeType: uploaded.mimeType ?? mimeType,
    name: uploaded.name,
    expiresAt,
  };
  fileRefMap.set(srcKey, ref);

  if (shapeCache) {
    const entry: CachedFilesApiEntry = {
      uri: ref.uri,
      mimeType: ref.mimeType,
      name: ref.name,
      expiresAt,
    };
    try {
      await shapeCache.writeFilesApi({ srcKey, entry });
    } catch { /* persistence failure non-fatal */ }
  }
}

function isFresh(expiresAt: number | undefined): boolean {
  if (typeof expiresAt !== "number") return false;
  return expiresAt > Date.now() + FILES_API_FRESHNESS_MARGIN_MS;
}

function deriveExpiresAt(uploaded: GoogleFile): number {
  const raw = (uploaded as { expirationTime?: unknown }).expirationTime;
  if (typeof raw === "string") {
    const parsed = Date.parse(raw);
    if (Number.isFinite(parsed)) return parsed;
  }
  return Date.now() + FILES_API_FALLBACK_TTL_MS;
}
