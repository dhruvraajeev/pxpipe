/**
 * Runtime-agnostic event sink for pixelpipe.
 *
 * The proxy core emits a `ProxyEvent` per request. A Tracker is the host's
 * decision about *where* those events go — local JSONL file on Node, console
 * (= Workers Logs) on Cloudflare. The shape of the persisted record is the
 * same on both sides so analysis tooling (`pixelpipe stats`, downstream
 * aggregation) doesn't care which runtime produced it.
 *
 * Privacy: we never emit raw user text or the system prompt. Only sizes,
 * counts, durations, parsed env fields (cwd / branch / platform), and short
 * sha256 prefixes. All callable from Node 18+ and Workers.
 */

import type { ProxyEvent } from './proxy.js';
import { bytesToBase64 } from './png.js';

/** The flat record shape that lands in JSONL / log lines. Adding a field
 *  here is a non-breaking change for readers. */
export interface TrackEvent {
  ts: string;
  method: string;
  path: string;
  status: number;
  duration_ms: number;
  first_byte_ms?: number;

  // From TransformInfo:
  compressed?: boolean;
  reason?: string;
  orig_chars?: number;
  /** Sum of text-chars actually replaced by image blocks this request (static
   *  slab + reminders + tool_results). Apples-to-apples with image_count for
   *  savings math: textTokens(compressed_chars/4) vs imageTokens(image_count×2500). */
  compressed_chars?: number;
  image_count?: number;
  image_bytes?: number;
  /** Total pixel area summed across all rendered images. Pairs with
   *  `cache_create_tokens` on cold-miss events for empirical px/token. */
  image_pixels?: number;
  /** Total TEXT chars remaining in the outgoing transformed body (every
   *  `text` block across system + messages, including tool_result text that
   *  didn't compress). Pairs with `image_pixels` so a regression over N
   *  cold-misses solves both `chars_per_token` (α) and `pixels_per_token` (β)
   *  for the live model. */
  outgoing_text_chars?: number;
  /** Ground-truth pre-transform token count from Anthropic's
   *  /v1/messages/count_tokens. Together with `actual_tokens_measured`
   *  this gives exact saved_pct without any α/β estimation. Absent on
   *  rows where the count_tokens upstream call failed. */
  baseline_tokens_measured?: number;
  /** Ground-truth post-transform token count from count_tokens. */
  actual_tokens_measured?: number;
  static_chars?: number;
  dynamic_chars?: number;
  dynamic_block_count?: number;
  /** Image count attributable to compressing `<system-reminder>` blocks in
   *  the first user message. */
  reminder_imgs?: number;
  /** Image count attributable to compressing tool_result content. */
  tool_result_imgs?: number;
  /** Number of tool_result blocks where the source text exceeded the
   *  per-tool_result image budget and was truncated before rendering. */
  truncated_tool_results?: number;
  /** Total chars elided by paging across all tool_results this request. */
  omitted_chars?: number;
  /** Variant C history-image: how many messages got collapsed into the
   *  prepended synthetic user message this request. Absent when not. */
  collapsed_turns?: number;
  /** Variant C: total chars of text serialized into the history image(s)
   *  before render. Absent when no collapse happened. */
  collapsed_chars?: number;
  /** Variant C: number of PNG image blocks emitted for the history. Folded
   *  into `image_count` too — surfaced separately so dashboards can
   *  attribute image-count growth to history vs system-slab vs reminders. */
  collapsed_images?: number;
  /** Variant C: why the history collapse didn't run (or did). Diagnostic. */
  history_reason?: string;
  /** Codepoints rendered into images that weren't in the glyph atlas. A
   *  spike here means users are typing glyphs we don't ship — consider
   *  switching ATLAS_PROFILE to `full-bmp`. */
  dropped_chars?: number;
  /** Top-20 dropped codepoints by frequency for this request, keyed
   *  `U+HHHH`. Only present when `dropped_chars > 0`. Lets the operator
   *  identify which Unicode blocks to add to the atlas profile without
   *  having to capture & inspect the request body. */
  dropped_codepoints_top?: Record<string, number>;
  /** Counters for blocks that DIDN'T get image-compressed this request. Lets
   *  the operator tune the break-even threshold and spot if the
   *  not-profitable bucket grows (= renderer-config change needed). Only
   *  emitted when at least one counter > 0. */
  passthrough_reasons?: { below_threshold?: number; not_profitable?: number };
  /** Tag names found in the static slab we don't recognize. Canary for
   *  Claude Code releases that add new dynamic tags. */
  unknown_static_tags?: string[];

  // From TransformInfo.env:
  cwd?: string;
  is_git_repo?: boolean;
  git_branch?: string;
  platform?: string;
  os_version?: string;
  today?: string;

  // Fingerprints:
  system_sha8?: string;
  claude_md_sha8?: string;
  first_user_sha8?: string;

  // From Anthropic Usage:
  input_tokens?: number;
  output_tokens?: number;
  cache_create_tokens?: number;
  cache_read_tokens?: number;

  // Errors:
  error?: string;
  /** First ~2 KiB of the upstream response body for 4xx requests. Lets us
   *  see what Anthropic actually rejected without re-running the request. */
  error_body?: string;
  /** sha256[0..8] of the TRANSFORMED outgoing request body. Set on every
   *  /v1/messages POST. Lets future debuggers correlate identical payloads
   *  across requests without persisting bodies. */
  req_body_sha8?: string;
  /** Gzipped+base64'd TRANSFORMED outgoing request body for 4xx requests,
   *  when it fits inline (≤ TRACK_BODY_INLINE_MAX after base64). The Node
   *  host redirects oversized bodies to a sidecar (see req_body_sample_path)
   *  before this serializer runs. */
  req_body_sample_b64?: string;
  /** Filesystem path to a gzipped sidecar copy of the TRANSFORMED outgoing
   *  request body, set by the Node host when the inline cap is exceeded.
   *  Workers never set this (no fs); they just drop the sample. */
  req_body_sample_path?: string;
}

/** Max base64-encoded length we'll inline in a single JSONL row. 32 KiB keeps
 *  the dashboard's per-row parse cost manageable while still holding the
 *  ~170 KiB raw bodies we've seen post-gzip+base64. Anything larger goes to
 *  a sidecar file (Node host) or gets dropped (Workers host). */
export const TRACK_BODY_INLINE_MAX = 32 * 1024;

/** Hosts implement this to persist events. */
export interface Tracker {
  emit(ev: TrackEvent): void | Promise<void>;
  /** Optional: flush any buffered writes (file rotation, etc.). */
  flush?(): void | Promise<void>;
}

/** Convert the in-memory ProxyEvent into the flat persisted shape. Lives in
 *  core so Node and Worker hosts can't drift from each other. */
export function toTrackEvent(ev: ProxyEvent): TrackEvent {
  const info = ev.info;
  const env = info?.env;
  const u = ev.usage;
  const out: TrackEvent = {
    ts: new Date().toISOString(),
    method: ev.method,
    path: ev.path,
    status: ev.status,
    duration_ms: ev.durationMs,
  };
  if (ev.firstByteMs !== undefined) out.first_byte_ms = ev.firstByteMs;
  if (ev.error) out.error = ev.error;
  if (ev.errorBody) out.error_body = ev.errorBody;
  if (ev.reqBodySha8) out.req_body_sha8 = ev.reqBodySha8;
  // Body sample: prefer the sidecar path (set by the Node host); else inline
  // the gzipped+base64'd body if it fits; else drop (Workers cap, or no body).
  if (ev.reqBodySamplePath) {
    out.req_body_sample_path = ev.reqBodySamplePath;
  } else if (ev.reqBodyGz && ev.reqBodyGz.byteLength > 0) {
    const b64 = bytesToBase64(ev.reqBodyGz);
    if (b64.length <= TRACK_BODY_INLINE_MAX) {
      out.req_body_sample_b64 = b64;
    }
    // else: too big and no sidecar — drop it. (Workers path; on Node the host
    // should have written the sidecar before this serializer ran.)
  }

  if (info) {
    if (info.compressed !== undefined) out.compressed = info.compressed;
    if (info.reason) out.reason = info.reason;
    if (info.origChars !== undefined) out.orig_chars = info.origChars;
    if (info.compressedChars !== undefined && info.compressedChars > 0) {
      out.compressed_chars = info.compressedChars;
    }
    if (info.imageCount !== undefined) out.image_count = info.imageCount;
    if (info.imageBytes !== undefined) out.image_bytes = info.imageBytes;
    if (info.imagePixels !== undefined && info.imagePixels > 0) {
      out.image_pixels = info.imagePixels;
    }
    if (info.outgoingTextChars !== undefined && info.outgoingTextChars > 0) {
      out.outgoing_text_chars = info.outgoingTextChars;
    }
    if (info.baselineTokensMeasured !== undefined && info.baselineTokensMeasured >= 0) {
      out.baseline_tokens_measured = info.baselineTokensMeasured;
    }
    if (info.actualTokensMeasured !== undefined && info.actualTokensMeasured >= 0) {
      out.actual_tokens_measured = info.actualTokensMeasured;
    }
    if (info.staticChars !== undefined) out.static_chars = info.staticChars;
    if (info.dynamicChars !== undefined) out.dynamic_chars = info.dynamicChars;
    if (info.dynamicBlockCount !== undefined) out.dynamic_block_count = info.dynamicBlockCount;
    if (info.reminderImgs !== undefined) out.reminder_imgs = info.reminderImgs;
    if (info.toolResultImgs !== undefined) out.tool_result_imgs = info.toolResultImgs;
    if (info.truncatedToolResults !== undefined && info.truncatedToolResults > 0) {
      out.truncated_tool_results = info.truncatedToolResults;
    }
    if (info.omittedChars !== undefined && info.omittedChars > 0) {
      out.omitted_chars = info.omittedChars;
    }
    if (info.collapsedTurns !== undefined && info.collapsedTurns > 0) {
      out.collapsed_turns = info.collapsedTurns;
    }
    if (info.collapsedChars !== undefined && info.collapsedChars > 0) {
      out.collapsed_chars = info.collapsedChars;
    }
    if (info.collapsedImages !== undefined && info.collapsedImages > 0) {
      out.collapsed_images = info.collapsedImages;
    }
    if (info.historyReason !== undefined) {
      out.history_reason = info.historyReason;
    }
    if (info.droppedChars !== undefined && info.droppedChars > 0) {
      out.dropped_chars = info.droppedChars;
    }
    if (info.droppedCodepointsTop && Object.keys(info.droppedCodepointsTop).length > 0) {
      out.dropped_codepoints_top = info.droppedCodepointsTop;
    }
    if (info.passthroughReasons) {
      const pr = info.passthroughReasons;
      if ((pr.below_threshold ?? 0) > 0 || (pr.not_profitable ?? 0) > 0) {
        out.passthrough_reasons = pr;
      }
    }
    if (info.unknownStaticTags && info.unknownStaticTags.length > 0)
      out.unknown_static_tags = info.unknownStaticTags;
    if (info.systemSha8) out.system_sha8 = info.systemSha8;
    if (info.claudeMdSha8) out.claude_md_sha8 = info.claudeMdSha8;
    if (info.firstUserSha8) out.first_user_sha8 = info.firstUserSha8;
  }
  if (env) {
    if (env.cwd) out.cwd = env.cwd;
    if (env.isGitRepo !== undefined) out.is_git_repo = env.isGitRepo;
    if (env.gitBranch) out.git_branch = env.gitBranch;
    if (env.platform) out.platform = env.platform;
    if (env.osVersion) out.os_version = env.osVersion;
    if (env.today) out.today = env.today;
  }
  if (u) {
    if (u.input_tokens !== undefined) out.input_tokens = u.input_tokens;
    if (u.output_tokens !== undefined) out.output_tokens = u.output_tokens;
    if (u.cache_creation_input_tokens !== undefined)
      out.cache_create_tokens = u.cache_creation_input_tokens;
    if (u.cache_read_input_tokens !== undefined)
      out.cache_read_tokens = u.cache_read_input_tokens;
  }
  return out;
}

/** Tracker that writes one JSON line per call to the given function. Used
 *  by the Worker host (sinkFn = console.log). The Node host uses a richer
 *  file-backed implementation that handles rotation. */
export class JsonLogTracker implements Tracker {
  constructor(private readonly sink: (line: string) => void = (s) => console.log(s)) {}
  emit(ev: TrackEvent): void {
    try {
      this.sink(JSON.stringify(ev));
    } catch {
      /* swallow — tracker must never break a request */
    }
  }
}

/** Tracker that drops everything. Used when PIXELPIPE_TRACK=0. */
export const noopTracker: Tracker = { emit() {} };
