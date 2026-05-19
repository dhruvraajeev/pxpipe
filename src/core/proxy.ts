/**
 * The pixelpipe proxy as a single Web-standard fetch handler.
 *
 * Both `src/node.ts` and `src/worker.ts` adapt this to their respective
 * runtimes (node:http server vs CF Worker `fetch` export). The handler
 * itself only uses `Request`, `Response`, `URL`, and global `fetch` — all
 * of which exist identically in Node 18+ and Workers.
 */

import { transformRequest, type TransformOptions, type TransformInfo } from './transform.js';
import type { Usage } from './types.js';

export interface ProxyConfig {
  /** Anthropic API base, no trailing slash. Defaults to api.anthropic.com. */
  upstream?: string;
  /** Override or supply an API key. If unset, we forward whatever the client sent. */
  apiKey?: string;
  /** Per-request transform options. Pass a function when the host wants to
   *  inject DYNAMIC values per request (e.g. live empirical `charsPerToken`
   *  from the dashboard's converging fit) — the proxy invokes it once per
   *  /v1/messages POST. Static object form is used by the Workers host and
   *  tests that don't need dynamic state. */
  transform?: TransformOptions | (() => TransformOptions);
  /** Called after every request — useful for logging / metrics in the host. */
  onRequest?: (event: ProxyEvent) => void | Promise<void>;
}

export interface ProxyEvent {
  method: string;
  path: string;
  status: number;
  /** Wall-clock ms from request start to event fire (≈ end of upstream response
   *  body, since we now wait for usage extraction). For first-byte latency see
   *  firstByteMs. */
  durationMs: number;
  /** Wall-clock ms from request start to upstream response headers. */
  firstByteMs?: number;
  info?: TransformInfo;
  /** Usage block from Anthropic's response — input/output/cache tokens. */
  usage?: Usage;
  error?: string;
  /** First ~2 KiB of the upstream response body when status is in [400, 499].
   *  Lets us see what Anthropic actually rejected without re-running the request.
   *  Not captured for 2xx (no error) or 5xx (we synthesize our own message). */
  errorBody?: string;
  /** sha256[0..8] of the TRANSFORMED outgoing request body. Set on every
   *  /v1/messages POST regardless of status. Lets future debuggers correlate
   *  "same payload, sometimes works, sometimes fails" without storing bodies. */
  reqBodySha8?: string;
  /** Full gzipped transformed body, populated only on 4xx. The Node host may
   *  redirect this to a sidecar file (see reqBodySamplePath) before the
   *  tracker serializes the event; Workers always inline-cap at 32 KiB. */
  reqBodyGz?: Uint8Array;
  /** Set by the Node host *in place of* reqBodyGz when it wrote the gzipped
   *  body to a sidecar file. The path lands in the JSONL as
   *  `req_body_sample_path`. */
  reqBodySamplePath?: string;
}

/** Max chars of upstream error body we surface on ProxyEvent. Keeps the JSONL
 *  line small while still being big enough to hold Anthropic's full error JSON
 *  (typically a few hundred bytes). */
const ERROR_BODY_MAX = 2048;

/** Gzip a byte buffer using the standard `CompressionStream`. Available in
 *  Node 18+ and Cloudflare Workers — no Buffer / no zlib. */
async function gzipBytes(body: Uint8Array): Promise<Uint8Array> {
  // `body as BufferSource`: TS doesn't model Response taking a Uint8Array
  // directly even though it works in both runtimes.
  const stream = new Response(body as BufferSource).body!.pipeThrough(
    new CompressionStream('gzip'),
  );
  const buf = await new Response(stream).arrayBuffer();
  return new Uint8Array(buf);
}

/** sha256[0..8] of a byte buffer, hex. Same shape as the existing sha8(text)
 *  helper in transform.ts but works on raw bytes (no extra encode pass). */
async function sha8Bytes(body: Uint8Array): Promise<string> {
  // Cast to BufferSource — Web Crypto accepts Uint8Array at runtime.
  const digest = await crypto.subtle.digest('SHA-256', body as BufferSource);
  const bytes = new Uint8Array(digest);
  let hex = '';
  for (let i = 0; i < 4; i++) hex += bytes[i]!.toString(16).padStart(2, '0');
  return hex;
}

/**
 * Tee the response body so we can scan for the usage block (SSE: in the
 * message_start event; non-stream: at the top of the JSON) without buffering
 * the whole stream or blocking the client. Returns the un-touched response
 * to forward to the client + a Promise that resolves to the parsed Usage
 * (or undefined if we couldn't find one within the budget).
 *
 * For upstream 4xx responses, we instead tee the body to capture up to
 * `ERROR_BODY_MAX` chars so the host can log what Anthropic actually rejected.
 * 5xx still bails — those get our own synthesized error string upstream.
 */
function teeForUsage(res: Response): {
  response: Response;
  usagePromise: Promise<Usage | undefined>;
  errorBodyPromise: Promise<string | undefined>;
} {
  // No body at all: nothing to extract on either path.
  if (!res.body) {
    return {
      response: res,
      usagePromise: Promise.resolve(undefined),
      errorBodyPromise: Promise.resolve(undefined),
    };
  }
  // 4xx: tee for the error body but skip usage scanning entirely.
  if (res.status >= 400 && res.status < 500) {
    const [forClient, forUs] = res.body.tee();
    const errorBodyPromise = (async (): Promise<string | undefined> => {
      const reader = forUs.getReader();
      const decoder = new TextDecoder();
      let out = '';
      try {
        while (out.length < ERROR_BODY_MAX) {
          const { done, value } = await reader.read();
          if (done) break;
          out += decoder.decode(value, { stream: true });
        }
        out += decoder.decode();
        // Drain the rest so the tee buffer doesn't hold the stream open.
        while (true) {
          const { done } = await reader.read();
          if (done) break;
        }
      } catch {
        /* client may have aborted; whatever we got is fine */
      }
      return out.length > ERROR_BODY_MAX ? out.slice(0, ERROR_BODY_MAX) : out;
    })();
    return {
      response: new Response(forClient, {
        status: res.status,
        statusText: res.statusText,
        headers: res.headers,
      }),
      usagePromise: Promise.resolve(undefined),
      errorBodyPromise,
    };
  }
  // 5xx: skip both (the host already synthesizes an error message).
  if (res.status >= 500) {
    return {
      response: res,
      usagePromise: Promise.resolve(undefined),
      errorBodyPromise: Promise.resolve(undefined),
    };
  }
  const ct = (res.headers.get('content-type') ?? '').toLowerCase();
  const [forClient, forUs] = res.body.tee();

  const usagePromise = (async (): Promise<Usage | undefined> => {
    const reader = forUs.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    const drain = async () => {
      try {
        while (true) {
          const { done } = await reader.read();
          if (done) break;
        }
      } catch {
        /* ignore */
      }
    };

    try {
      if (ct.includes('text/event-stream')) {
        // SSE: usage is in the FIRST event (`message_start`). Cap scan at 64
        // KiB so we don't hold the tee buffer open for the entire stream.
        const MAX = 65536;
        while (buf.length < MAX) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const idx = buf.indexOf('event: message_start');
          if (idx >= 0) {
            // The data: line follows. Match the first data: after that idx.
            const m = /^data:\s*(.+)$/m.exec(buf.slice(idx));
            if (m) {
              try {
                const j = JSON.parse(m[1]!);
                void drain();
                return j?.message?.usage as Usage | undefined;
              } catch {
                /* not yet a complete JSON line — keep reading */
              }
            }
          }
        }
        void drain();
        return undefined;
      }

      if (ct.includes('application/json')) {
        // Non-stream: buffer fully (capped at 4 MiB).
        const MAX = 4 * 1024 * 1024;
        while (buf.length < MAX) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
        }
        try {
          const j = JSON.parse(buf);
          return j?.usage as Usage | undefined;
        } catch {
          return undefined;
        }
      }
    } catch {
      /* tee may be released early if the client aborts — ignore */
    }
    void drain();
    return undefined;
  })();

  return {
    response: new Response(forClient, {
      status: res.status,
      statusText: res.statusText,
      headers: res.headers,
    }),
    usagePromise,
    errorBodyPromise: Promise.resolve(undefined),
  };
}

const DEFAULT_UPSTREAM = 'https://api.anthropic.com';

/** Headers we strip on the way out — they're hop-by-hop or proxy-injected. */
const STRIP_REQ_HEADERS = new Set([
  'host',
  'connection',
  'keep-alive',
  'proxy-connection',
  'transfer-encoding',
  'upgrade',
  'content-length', // we recompute
  'expect',
  'accept-encoding', // let upstream choose
]);

const STRIP_RES_HEADERS = new Set([
  'connection',
  'keep-alive',
  'transfer-encoding',
  'content-encoding', // we don't re-encode
  'content-length',   // body may differ after streaming
]);

function filterHeaders(src: Headers, strip: Set<string>): Headers {
  const out = new Headers();
  src.forEach((v, k) => {
    if (!strip.has(k.toLowerCase())) out.append(k, v);
  });
  return out;
}

/** Ask the upstream /v1/messages/count_tokens endpoint to tokenize a body
 *  using the same auth + headers we'd send to /v1/messages. Returns the
 *  exact input_tokens count or null on any failure (4xx, 5xx, network
 *  error, missing field). count_tokens is documented as free — Anthropic
 *  does not bill input tokens for it — so we use it to measure ground-
 *  truth pre/post-transform sizes without estimation.
 *
 *  Failure is never fatal: when this returns null the caller skips the
 *  measurement and the dashboard falls back to the regression estimate. */
async function countTokensUpstream(
  upstream: string,
  body: BodyInit,
  headers: Headers,
): Promise<number | null> {
  try {
    const res = await fetch(upstream + '/v1/messages/count_tokens', {
      method: 'POST',
      headers,
      body,
      ...(body instanceof ReadableStream ? { duplex: 'half' } : {}),
    } as RequestInit);
    if (!res.ok) return null;
    const text = await res.text();
    const json = JSON.parse(text) as { input_tokens?: unknown };
    return typeof json.input_tokens === 'number' ? json.input_tokens : null;
  } catch {
    return null;
  }
}

/** Build the proxy fetch handler bound to a config. */
export function createProxy(config: ProxyConfig = {}) {
  const upstream = (config.upstream ?? DEFAULT_UPSTREAM).replace(/\/+$/, '');

  return async function handle(req: Request): Promise<Response> {
    const t0 = Date.now();
    const url = new URL(req.url);
    const path = url.pathname + url.search;

    // Captured during the transform step. `reqBodyBytes` is the raw
    // transformed body — kept around so we can gzip it lazily on 4xx without
    // having to re-stringify. `reqBodySha8` is computed eagerly because
    // it's cheap and lands on every event (4xx and 2xx) for correlation.
    let reqBodyBytes: Uint8Array | undefined;
    let reqBodySha8: string | undefined;

    const fire = (
      status: number,
      info?: TransformInfo,
      error?: string,
      firstByteMs?: number,
      usage?: Usage,
      errorBody?: string,
    ): void => {
      const is4xx = status >= 400 && status < 500;
      // Gzip the full body only when we actually need it — i.e. status is 4xx
      // and we have bytes to capture. Awaiting inside an async IIFE keeps the
      // fire() signature unchanged; the host receives the event once the
      // gzip resolves (or immediately if not 4xx).
      const finalize = async (): Promise<void> => {
        let reqBodyGz: Uint8Array | undefined;
        if (is4xx && reqBodyBytes && reqBodyBytes.byteLength > 0) {
          try {
            reqBodyGz = await gzipBytes(reqBodyBytes);
          } catch {
            // gzip failure is non-fatal — drop the body sample, keep the rest.
          }
        }
        // If count_tokens measurement was kicked off in parallel, await it
        // here so the resulting numbers land on `info` BEFORE the host's
        // onRequest fires (which is what persists the event). null results
        // (measurement failed) are silently dropped — the dashboard falls
        // back to the α-regression estimate when these fields are absent.
        if (measurePromise && info) {
          try {
            const m = await measurePromise;
            if (m.baseline !== null) info.baselineTokensMeasured = m.baseline;
            if (m.actual !== null) info.actualTokensMeasured = m.actual;
          } catch {
            // measurement failed — drop, keep the rest of the event intact.
          }
        }
        await config.onRequest?.({
          method: req.method,
          path: url.pathname,
          status,
          durationMs: Date.now() - t0,
          firstByteMs,
          info,
          usage,
          error,
          errorBody,
          reqBodySha8,
          reqBodyGz,
        });
      };
      void finalize();
    };

    // Only intercept /v1/messages POSTs. Everything else passes through.
    const isMessages = req.method === 'POST' && url.pathname === '/v1/messages';

    let bodyOut: BodyInit | null = null;
    let info: TransformInfo | undefined;

    // Ground-truth token-count measurement. Fires /v1/messages/count_tokens
    // on the pre-transform and post-transform bodies in parallel with the
    // main upstream forward. Results land on info.{baselineTokensMeasured,
    // actualTokensMeasured} and the dashboard reports the real saved_pct
    // from these exact numbers — no α/β estimation.
    let measurePromise: Promise<{
      baseline: number | null;
      actual: number | null;
    }> | undefined;

    if (isMessages) {
      const bodyIn = new Uint8Array(await req.arrayBuffer());
      try {
        // Resolve transform options per-request when the host passed a
        // function — lets dashboardState.fitCosts()'s live α flow into
        // isCompressionProfitable on every call.
        const transformOpts =
          typeof config.transform === 'function' ? config.transform() : config.transform;
        const r = await transformRequest(bodyIn, transformOpts);
        // Cast: TS narrows Uint8Array<ArrayBufferLike> away from BodyInit, but
        // it's a valid body and we never use SharedArrayBuffer.
        bodyOut = r.body as unknown as BodyInit;
        info = r.info;
        // Stash the raw bytes and eagerly hash them. Hash lands on every event
        // (cheap, ~ a SHA-256 over a few hundred KB). The bytes themselves are
        // only gzipped+emitted on 4xx — see `fire`.
        reqBodyBytes = r.body;
        if (r.body.byteLength > 0) {
          reqBodySha8 = await sha8Bytes(r.body);
        }

        // Kick off the two count_tokens probes BEFORE forwarding /v1/messages
        // so they run in parallel with the main request. Headers are filtered
        // exactly like the main request — same auth, same model, same
        // anthropic-version. Failure is silent: a null result drops the
        // measurement on this event without affecting the forwarded response.
        const ctHeaders = filterHeaders(req.headers, STRIP_REQ_HEADERS);
        if (config.apiKey) ctHeaders.set('x-api-key', config.apiKey);
        measurePromise = Promise.all([
          countTokensUpstream(upstream, bodyIn, ctHeaders),
          countTokensUpstream(upstream, r.body as unknown as BodyInit, ctHeaders),
        ]).then(([baseline, actual]) => ({ baseline, actual }));
      } catch (e) {
        fire(502, undefined, `transform_error: ${(e as Error).message}`);
        return new Response(JSON.stringify({ error: 'pixelpipe transform failed' }), {
          status: 502,
          headers: { 'content-type': 'application/json' },
        });
      }
    } else {
      // Pass body through unchanged.
      bodyOut = req.body;
    }

    const outHeaders = filterHeaders(req.headers, STRIP_REQ_HEADERS);
    if (config.apiKey) outHeaders.set('x-api-key', config.apiKey);

    const upstreamUrl = upstream + path;
    let upstreamRes: Response;
    try {
      upstreamRes = await fetch(upstreamUrl, {
        method: req.method,
        headers: outHeaders,
        body: bodyOut,
        // duplex is required by spec when sending a stream as body
        ...(bodyOut instanceof ReadableStream ? { duplex: 'half' } : {}),
      } as RequestInit);
    } catch (e) {
      fire(502, info, `upstream_error: ${(e as Error).message}`);
      return new Response(JSON.stringify({ error: 'pixelpipe upstream unreachable' }), {
        status: 502,
        headers: { 'content-type': 'application/json' },
      });
    }

    const firstByteMs = Date.now() - t0;

    // Tee the upstream body so we can extract Anthropic's usage block. The
    // client gets one side immediately; we read the other in the background.
    // For 4xx responses we also tee to capture the error body (up to 2 KiB)
    // so the host can log what Anthropic actually rejected.
    const { response: teed, usagePromise, errorBodyPromise } = teeForUsage(upstreamRes);

    // Fire the host event once usage AND any captured error body are known
    // (or once we've given up on finding them). Don't await — the response
    // below is what unblocks the client; fire happens in the background.
    void Promise.all([
      usagePromise.catch(() => undefined),
      errorBodyPromise.catch(() => undefined),
    ]).then(([usage, errorBody]) =>
      fire(upstreamRes.status, info, undefined, firstByteMs, usage, errorBody),
    );

    return new Response(teed.body, {
      status: upstreamRes.status,
      statusText: upstreamRes.statusText,
      headers: filterHeaders(upstreamRes.headers, STRIP_RES_HEADERS),
    });
  };
}
