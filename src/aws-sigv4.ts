/**
 * AWS Signature V4 — minimal static-credential signer for Bedrock.
 *
 * Scope:
 *  - Static credentials only (accessKeyId + secretAccessKey, optional sessionToken).
 *    No IAM role assume, no IMDS, no SSO, no `~/.aws/credentials` file reads.
 *  - Request-payload signing only (single-shot SHA256 of body). Bedrock's
 *    `invoke-with-response-stream` puts the entire body in one POST and only
 *    the *response* streams — so streaming-payload chunk signing
 *    (`STREAMING-AWS4-HMAC-SHA256-PAYLOAD`, used by S3 PUT) is NOT needed.
 *  - Single-pass URI encoding for the canonical path (suitable for `bedrock`,
 *    not S3 — S3 requires double-encoding outside this signer).
 *
 * Reference: https://docs.aws.amazon.com/general/latest/gr/sigv4_signing.html
 */

import { createHash, createHmac } from "node:crypto";

export interface SigV4Credentials {
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken?: string;
}

export interface SigV4SignInput {
  method: "POST" | "GET";
  /** Full URL — host / path / query are extracted internally. */
  url: string;
  region: string;
  /** AWS service id (e.g. "bedrock"). Lowercase. */
  service: string;
  /** Already-serialized request body (use empty string for GET). */
  body: string;
  /** Caller-provided headers. `host` will be added automatically; do NOT set it. */
  headers: Record<string, string>;
  /** Inject a fixed timestamp for deterministic test vectors. */
  now?: Date;
}

const ALGORITHM = "AWS4-HMAC-SHA256";

function sha256Hex(data: string | Buffer): string {
  return createHash("sha256").update(data).digest("hex");
}

function hmac(key: Buffer | string, data: string): Buffer {
  return createHmac("sha256", key).update(data).digest();
}

/** RFC3986 unreserved-only encoding (AWS-canonical: encodeURIComponent + extra
 *  guards on `!` `'` `(` `)` `*`). */
function uriEncode(s: string, encodeSlash = true): string {
  let out = "";
  for (const ch of s) {
    const code = ch.charCodeAt(0);
    const isUnreserved =
      (code >= 0x41 && code <= 0x5a) || // A-Z
      (code >= 0x61 && code <= 0x7a) || // a-z
      (code >= 0x30 && code <= 0x39) || // 0-9
      ch === "-" || ch === "_" || ch === "." || ch === "~";
    if (isUnreserved) {
      out += ch;
    } else if (ch === "/" && !encodeSlash) {
      out += ch;
    } else {
      out += [...Buffer.from(ch, "utf-8")]
        .map((b) => "%" + b.toString(16).toUpperCase().padStart(2, "0"))
        .join("");
    }
  }
  return out;
}

function formatAmzDate(d: Date): { amzDate: string; dateStamp: string } {
  // amzDate: YYYYMMDDTHHMMSSZ ; dateStamp: YYYYMMDD
  const iso = d.toISOString();
  const amzDate = iso.replace(/[:-]/g, "").replace(/\.\d{3}Z$/, "Z");
  const dateStamp = amzDate.slice(0, 8);
  return { amzDate, dateStamp };
}

function canonicalQueryString(search: string): string {
  if (!search || search === "?") return "";
  const q = search.startsWith("?") ? search.slice(1) : search;
  const params: Array<[string, string]> = [];
  for (const part of q.split("&")) {
    if (!part) continue;
    const eq = part.indexOf("=");
    const k = eq === -1 ? part : part.slice(0, eq);
    const v = eq === -1 ? "" : part.slice(eq + 1);
    // Decode caller input first (URL constructor leaves it pct-encoded), then
    // re-encode using AWS canonical rules so we don't accidentally double-encode.
    params.push([uriEncode(decodeURIComponent(k)), uriEncode(decodeURIComponent(v))]);
  }
  params.sort(([ak, av], [bk, bv]) => (ak < bk ? -1 : ak > bk ? 1 : av < bv ? -1 : av > bv ? 1 : 0));
  return params.map(([k, v]) => `${k}=${v}`).join("&");
}

function canonicalHeaders(headers: Record<string, string>): {
  canonical: string;
  signed: string;
} {
  // Lowercase keys, trim values' inner whitespace per spec.
  const entries: Array<[string, string]> = Object.entries(headers).map(([k, v]) => [
    k.toLowerCase(),
    v.trim().replace(/\s+/g, " "),
  ]);
  entries.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  const canonical = entries.map(([k, v]) => `${k}:${v}`).join("\n") + "\n";
  const signed = entries.map(([k]) => k).join(";");
  return { canonical, signed };
}

/**
 * Sign a request and return the FINAL header set (input headers + host +
 * x-amz-date + x-amz-content-sha256 + optional x-amz-security-token + Authorization).
 * Caller passes the returned headers verbatim to fetch.
 */
export function sigv4Sign(
  creds: SigV4Credentials,
  input: SigV4SignInput,
): Record<string, string> {
  const u = new URL(input.url);
  const host = u.host;
  const path = u.pathname || "/";
  const canonicalPath = uriEncode(path, false); // do NOT encode `/` in path
  const canonicalQuery = canonicalQueryString(u.search);

  const now = input.now ?? new Date();
  const { amzDate, dateStamp } = formatAmzDate(now);

  const payloadHash = sha256Hex(input.body ?? "");

  const headers: Record<string, string> = {
    ...input.headers,
    host,
    "x-amz-date": amzDate,
    "x-amz-content-sha256": payloadHash,
  };
  if (creds.sessionToken) {
    headers["x-amz-security-token"] = creds.sessionToken;
  }

  const { canonical: canonicalHeaderBlock, signed: signedHeaders } =
    canonicalHeaders(headers);

  const canonicalRequest = [
    input.method,
    canonicalPath,
    canonicalQuery,
    canonicalHeaderBlock,
    signedHeaders,
    payloadHash,
  ].join("\n");

  const credentialScope = `${dateStamp}/${input.region}/${input.service}/aws4_request`;
  const stringToSign = [
    ALGORITHM,
    amzDate,
    credentialScope,
    sha256Hex(canonicalRequest),
  ].join("\n");

  const kDate = hmac("AWS4" + creds.secretAccessKey, dateStamp);
  const kRegion = hmac(kDate, input.region);
  const kService = hmac(kRegion, input.service);
  const kSigning = hmac(kService, "aws4_request");
  const signature = createHmac("sha256", kSigning).update(stringToSign).digest("hex");

  const authorization =
    `${ALGORITHM} Credential=${creds.accessKeyId}/${credentialScope}, ` +
    `SignedHeaders=${signedHeaders}, Signature=${signature}`;

  return { ...headers, Authorization: authorization };
}
