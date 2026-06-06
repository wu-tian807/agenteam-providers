// External media-byte readers. Providers stay sandbox-agnostic by declaring
// this narrow interface; the consumer (a host application) supplies an
// implementation as part of `ProviderDeps.readers` when calling
// `createProvider()`. Pure host-fs paths are served by the package-default
// implementation (`defaultMediaReaders` below, a thin async wrapper around
// `node:fs.readFile`); container/sandbox paths typically need a custom
// implementation that proxies the read through `docker exec` or some other
// bridge.
//
// This module is types-only at the contract level — no module-level state.
// Each `createProvider()` call carries its own readers through the deps
// object, captured by the raw adapter at construction time. That keeps test
// setup trivial (just pass a mock readers object) and lets multiple host
// applications coexist in one process without stepping on each other.

import { readFile } from "node:fs/promises";
import type { ContentPart } from "@agenteam/types";
import { coerceMimeBySniff } from "./media-mime-sniff.js";

export type MediaInputPart = Extract<ContentPart, {
  type: "image" | "image_file" | "audio" | "audio_file" | "video" | "video_file";
}>;

export type FilePathPart = Extract<ContentPart, { path: string }>;

export interface ReadResult {
  bytes: Buffer;
  mimeType: string;
  label: string;
}

export interface MediaReaders {
  /** Read media bytes (image/audio/video — file or inline). Returns
   *  sniff-corrected mime so a `.png` declaring `image/png` but actually
   *  carrying JPEG bytes gets the corrected `image/jpeg` here. */
  readMediaBytes(part: MediaInputPart): Promise<ReadResult>;

  /** Read arbitrary file bytes (text_file / file / *_file). Same path
   *  ownership rules as `readMediaBytes` — `inContainer === false` ⇒ host fs,
   *  otherwise the consumer's sandbox bridge (if they wired one in). */
  readFileBytes(part: FilePathPart): Promise<ReadResult>;
}

/**
 * Default `MediaReaders` — pure async `node:fs.readFile` for path-based parts,
 * inline base64 decode for inline parts, mime sniff applied to both. Suitable
 * for any consumer whose model inputs reference real host filesystem paths
 * (CLI tools, scripts, plain server processes). Hosts that need a sandbox /
 * container bridge ship their own implementation and pass it via `deps.readers`.
 *
 * The `inContainer` flag is ignored here — there's no bridge to route through
 * — every path resolves to a host-fs read. A consumer that uses
 * `inContainer === true` paths MUST replace this default.
 */
export const defaultMediaReaders: MediaReaders = {
  async readMediaBytes(part) {
    if ("path" in part) {
      const bytes = await readFile(part.path);
      const mimeType = coerceMimeBySniff(bytes, pickMime(part), part.path);
      return { bytes, mimeType, label: part.path };
    }
    const bytes = Buffer.from(part.data, "base64");
    const label = `inline ${part.type}`;
    const mimeType = coerceMimeBySniff(bytes, pickMime(part), label);
    return { bytes, mimeType, label };
  },
  async readFileBytes(part) {
    const bytes = await readFile(part.path);
    const mimeType = coerceMimeBySniff(bytes, pickMime(part), part.path);
    return { bytes, mimeType, label: part.path };
  },
};

// `mimeType` is only present on a subset of `ContentPart` variants — TS won't
// expose it as a property of the union. Centralise the safe pluck here so
// readers stay free of inline `as { mimeType?: string }` casts.
function pickMime(part: object): string | undefined {
  const mime = (part as { mimeType?: unknown }).mimeType;
  return typeof mime === "string" ? mime : undefined;
}
