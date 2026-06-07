/**
 * AWS event-stream binary frame parser — minimal Bedrock-focused decoder.
 *
 * Bedrock `invoke-with-response-stream` returns the upstream provider's SSE
 * chunks (here: Anthropic Messages API events) wrapped in AWS event-stream
 * frames. Each frame:
 *
 *   [total-len 4B BE]
 *   [headers-len 4B BE]
 *   [prelude-CRC32 4B BE]    (← intentionally NOT verified — see below)
 *   [headers... ]            (variable, sum = headers-len)
 *   [payload... ]            (variable, sum = total-len - headers-len - 16)
 *   [message-CRC32 4B BE]    (← intentionally NOT verified)
 *
 * Each header entry:
 *   [name-len 1B][name bytes][value-type 1B][value-len 2B BE][value bytes]
 *
 * For Bedrock the payload is a JSON `{"bytes": "<base64 of chunk JSON>"}` and
 * the headers carry `:event-type`, `:message-type`, `:content-type`,
 * `:exception-type` (when message-type=exception).
 *
 * Why no CRC verification: TLS already gives us byte integrity end-to-end,
 * AWS itself never delivers truncated frames in practice, and pulling in a
 * CRC32 implementation just to revalidate what TLS already covers buys us
 * nothing on this code path. If a frame is structurally malformed (length
 * mismatch, JSON parse failure) we still throw — we just don't compute CRCs.
 */

const HEADER_VAL_STRING = 7; // the only header value type Bedrock emits in practice

interface RawFrame {
  headers: Record<string, string>;
  payload: Buffer;
}

/**
 * Public yield shape — matches `parseSSE`'s `{event, data}` interface so
 * `aws-bedrock-anthropic.ts` can reuse the same event-loop body as
 * `anthropic.ts`. `data` is already JSON-parsed (saves the consumer a try/catch).
 */
export interface AwsEventChunk {
  event: string;
  data: any;
}

class FrameAccumulator {
  private buf: Buffer = Buffer.alloc(0);

  push(chunk: Buffer): void {
    this.buf = this.buf.length === 0 ? chunk : Buffer.concat([this.buf, chunk]);
  }

  /** Try to read one complete frame; returns null if buffer is short. */
  tryRead(): RawFrame | null {
    if (this.buf.length < 12) return null; // need at least the prelude
    const totalLen = this.buf.readUInt32BE(0);
    if (this.buf.length < totalLen) return null;

    const headersLen = this.buf.readUInt32BE(4);
    // prelude CRC at [8..12) — ignored
    const headersStart = 12;
    const headersEnd = headersStart + headersLen;
    const payloadEnd = totalLen - 4; // last 4B is message CRC, ignored

    if (headersEnd > payloadEnd) {
      throw new Error(
        `aws-eventstream: malformed frame — headers extend past payload boundary ` +
        `(totalLen=${totalLen}, headersLen=${headersLen})`,
      );
    }

    const headers = parseHeaders(this.buf.subarray(headersStart, headersEnd));
    const payload = Buffer.from(this.buf.subarray(headersEnd, payloadEnd));

    this.buf = Buffer.from(this.buf.subarray(totalLen));
    return { headers, payload };
  }
}

function parseHeaders(buf: Buffer): Record<string, string> {
  const out: Record<string, string> = {};
  let i = 0;
  while (i < buf.length) {
    const nameLen = buf.readUInt8(i); i += 1;
    const name = buf.subarray(i, i + nameLen).toString("utf-8"); i += nameLen;
    const valType = buf.readUInt8(i); i += 1;
    if (valType !== HEADER_VAL_STRING) {
      // Skip unknown types defensively — Bedrock only emits string headers,
      // but we don't want a future protocol bump to crash the loop. Read the
      // length+value blindly and move on.
      const valLen = buf.readUInt16BE(i); i += 2;
      i += valLen;
      continue;
    }
    const valLen = buf.readUInt16BE(i); i += 2;
    const value = buf.subarray(i, i + valLen).toString("utf-8"); i += valLen;
    out[name] = value;
  }
  return out;
}

/**
 * Bedrock-specific frame translation. The AWS frame's headers tell us what
 * KIND of event it is; the payload is JSON `{bytes: base64(...)}` whose decoded
 * body is the actual upstream SSE chunk we want. This function returns the
 * decoded chunk JSON paired with the header-derived event name, ready to feed
 * into the same SSE consumer that `anthropic.ts` already uses.
 *
 * Throws on:
 *   - exception frames (`:message-type: exception`) — converts the structured
 *     error body into a thrown Error so the fallback chain can re-classify it.
 *   - malformed JSON in the inner `bytes` blob.
 */
function frameToChunk(frame: RawFrame): AwsEventChunk {
  const messageType = frame.headers[":message-type"];
  const eventType = frame.headers[":event-type"];

  if (messageType === "exception" || messageType === "error") {
    const errBody = frame.payload.length
      ? safeJson(frame.payload.toString("utf-8"))
      : { message: messageType };
    const exType = frame.headers[":exception-type"] ?? frame.headers[":error-code"] ?? "unknown";
    const msg = errBody?.message ?? errBody?.Message ?? JSON.stringify(errBody);
    const err = new Error(`Bedrock ${exType}: ${msg}`);
    (err as any).bedrockException = exType;
    (err as any).bedrockBody = errBody;
    throw err;
  }

  if (frame.payload.length === 0) {
    return { event: eventType ?? "unknown", data: null };
  }

  const outer = safeJson(frame.payload.toString("utf-8"));
  if (outer && typeof outer.bytes === "string") {
    const innerJson = Buffer.from(outer.bytes, "base64").toString("utf-8");
    return { event: eventType ?? "unknown", data: safeJson(innerJson) };
  }
  // Some Bedrock event types deliver inline JSON without the `bytes` wrapper —
  // pass them through unchanged.
  return { event: eventType ?? "unknown", data: outer };
}

function safeJson(s: string): any {
  try { return JSON.parse(s); } catch {
    throw new Error(`aws-eventstream: invalid JSON payload: ${s.slice(0, 256)}`);
  }
}

/**
 * Consume a Bedrock event-stream HTTP response and yield decoded chunks in
 * the same `{event, data}` shape that `parseSSE` produces. Each `data` is
 * already JSON-parsed. Exception frames throw, ending the stream — callers
 * can let the error propagate to the fallback chain.
 */
export async function* parseAwsEventStream(
  res: Response,
): AsyncGenerator<AwsEventChunk, void, void> {
  if (!res.body) return;
  const reader = res.body.getReader();
  const acc = new FrameAccumulator();

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value || value.length === 0) continue;
      acc.push(Buffer.from(value));

      // Drain every complete frame currently buffered. A single TCP read can
      // contain multiple small frames or a partial large one — both handled.
      for (;;) {
        const frame = acc.tryRead();
        if (!frame) break;
        yield frameToChunk(frame);
      }
    }
  } finally {
    try { reader.releaseLock(); } catch { /* ignore */ }
  }
}

// ── Test helpers (not part of the runtime API) ───────────────────────────

/**
 * Encode a single AWS event-stream frame from headers + payload. Used by
 * tests to construct fixtures. CRC fields are zero-filled (parser ignores them).
 */
export function encodeAwsEventFrame(
  headers: Record<string, string>,
  payload: Buffer,
): Buffer {
  const headerParts: Buffer[] = [];
  for (const [k, v] of Object.entries(headers)) {
    const nameBuf = Buffer.from(k, "utf-8");
    const valBuf = Buffer.from(v, "utf-8");
    const part = Buffer.alloc(1 + nameBuf.length + 1 + 2 + valBuf.length);
    let p = 0;
    part.writeUInt8(nameBuf.length, p); p += 1;
    nameBuf.copy(part, p); p += nameBuf.length;
    part.writeUInt8(HEADER_VAL_STRING, p); p += 1;
    part.writeUInt16BE(valBuf.length, p); p += 2;
    valBuf.copy(part, p);
    headerParts.push(part);
  }
  const headersBuf = Buffer.concat(headerParts);
  const totalLen = 12 + headersBuf.length + payload.length + 4;
  const out = Buffer.alloc(totalLen);
  out.writeUInt32BE(totalLen, 0);
  out.writeUInt32BE(headersBuf.length, 4);
  // prelude CRC [8..12) and message CRC [end-4..end] left as zeros (ignored)
  headersBuf.copy(out, 12);
  payload.copy(out, 12 + headersBuf.length);
  return out;
}
