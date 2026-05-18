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
  image_count?: number;
  image_bytes?: number;
  static_chars?: number;
  dynamic_chars?: number;
  dynamic_block_count?: number;
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
}

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

  if (info) {
    if (info.compressed !== undefined) out.compressed = info.compressed;
    if (info.reason) out.reason = info.reason;
    if (info.origChars !== undefined) out.orig_chars = info.origChars;
    if (info.imageCount !== undefined) out.image_count = info.imageCount;
    if (info.imageBytes !== undefined) out.image_bytes = info.imageBytes;
    if (info.staticChars !== undefined) out.static_chars = info.staticChars;
    if (info.dynamicChars !== undefined) out.dynamic_chars = info.dynamicChars;
    if (info.dynamicBlockCount !== undefined) out.dynamic_block_count = info.dynamicBlockCount;
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
