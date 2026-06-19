/**
 * Request-body transformer. Extracts the static system prompt + tool definitions,
 * renders them as PNG image blocks, and rewrites the body to reference those images —
 * saving 65-73% input tokens while preserving reasoning quality.
 */

import type {
  ContentBlock,
  ImageBlock,
  Message,
  MessagesRequest,
  SystemField,
  TextBlock,
  ToolDef,
  ToolResultBlock,
  ToolUseBlock,
} from './types.js';
import {
  renderTextToPngs,
  renderTextToPngsMultiCol,
  reflow,
  maxFittingCols,
  shrinkColsToContent,
  MAX_HEIGHT_PX,
  NL_SENTINEL,
  PAD_X,
  PAD_Y,
  CELL_W,
  CELL_H,
  READABLE_CHARS_PER_IMAGE,
  DENSE_CONTENT_CHARS_PER_IMAGE,
  DENSE_CONTENT_COLS,
  DENSE_RENDER_STYLE,
  renderTextToPngsWithCharLimit,
} from './render.js';
import { bytesToBase64 } from './png.js';
import { collapseHistory } from './history.js';
import { CACHE_CREATE_RATE, CACHE_READ_RATE } from './baseline.js';

/** Per-block descriptor passed to `TransformOptions.keepSharp`. */
export interface KeepSharpBlock {
  /** Which live-region path is asking: `reminder`, `tool_result`, or `tool_result_part`. */
  readonly kind: 'reminder' | 'tool_result' | 'tool_result_part';
  /** The block's text exactly as the caller produced it (pre-render, pre-compaction). */
  readonly text: string;
  /** `tool_use_id` of the owning tool_result, when applicable. */
  readonly toolUseId?: string;
}

/** A block pxpipe rendered to image(s), returned in `TransformInfo.recoverable`
 *  when the caller sets `emitRecoverable`. Lets a stateful harness restore
 *  byte-exact content if the model needs the imaged region verbatim. */
export interface RecoverableBlock {
  /** `rec_` + 8 hex SHA-256 over kind + toolUseId + original text. */
  readonly id: string;
  readonly kind: 'reminder' | 'tool_result' | 'tool_result_part';
  readonly toolUseId?: string;
  /** Original text before compaction/reflow/paging — the bytes to restore. */
  readonly text: string;
  readonly imageCount: number;
}

export interface TransformOptions {
  /** Master switch — false makes this a no-op pass-through. */
  compress?: boolean;
  /** Move tool descriptions into the same image (and stub the originals). */
  compressTools?: boolean;
  /** Include full input_schema JSON for each tool. */
  compressSchemas?: boolean;
  /** Compress large `<system-reminder>` text blocks in the first user message. */
  compressReminders?: boolean;
  /** Compress large tool_result text content across all user messages. */
  compressToolResults?: boolean;
  /** Don't compress if total compressible chars below this. */
  minCompressChars?: number;
  /** Per-block threshold for compressReminders (chars). */
  minReminderChars?: number;
  /** Per-block threshold for compressToolResults (chars). */
  minToolResultChars?: number;
  /** Soft-wrap column count. */
  cols?: number;
  /** Hard upper bound on images per tool_result; source text truncated with a paging
   *  marker above this to stay under Anthropic's 100-image/request cap. Default 10. */
  maxImagesPerToolResult?: number;
  /** Pack N text columns side-by-side per image. Default 1. Auto-clamped to stay
   *  under 2000 px wide. OCR ordering risk at N≥2: model must read col 1 top-to-bottom
   *  before col 2. */
  multiCol?: number;
  /** Chars-per-token assumption for `isCompressionProfitable()`. Default 4. */
  charsPerToken?: number;
  /** Multi-turn amortization horizon for the history-collapse gate. N≥2 evaluates as
   *  if N future turns share the prefix (worst-case-warm-image vs best-case-warm-text).
   *  Default 1 (per-turn cold gate). See docs/HISTORY_CACHE_MODEL.md. */
  historyAmortizationHorizon?: number;
  /** Tokens the un-rewritten path would have cache-hit on. Adds a one-time burn
   *  penalty `priorWarmTokens × (CC − CR)` to the image side so the gate accounts
   *  for invalidating a warm text cache. Default 0 (cold-start). ≤0 clamped to 0. */
  priorWarmTokens?: number;
  /** Symmetric counterpart: tokens the image path would have cache-hit on. Adds the
   *  same burn formula to the TEXT side, preventing the gate from flipping out of
   *  image mode when the image prefix is already warm. Default 0. ≤0 clamped to 0. */
  priorWarmImageTokens?: number;
  /** Re-pack image-bound text into a ↵-delimited stream to fill `cols` (~29%→75-80%
   *  glyph-fill). ON by default (98.95% char accuracy at L1 OCR eval, +1pp vs baseline).
   *  Hard newlines become visible ↵ glyphs — tell the model via system prompt. */
  reflow?: boolean;
  /** Caller fidelity hint: return `true` for a block that must stay as text (IDs,
   *  hashes, file paths — content where mis-OCR would be silent and wrong). Only
   *  consulted on per-block live-region paths (reminders, tool_results). A throwing
   *  or non-boolean return is treated as `false`. */
  keepSharp?: (block: KeepSharpBlock) => boolean;
  /** When true, populate `TransformInfo.recoverable` with original text + provenance
   *  for every block rendered to images. Off by default (entries inflate `info`;
   *  only a stateful harness can use them). */
  emitRecoverable?: boolean;
}

const DEFAULTS: Required<TransformOptions> = {
  compress: true,
  compressTools: true,
  compressSchemas: true,
  compressReminders: true,
  compressToolResults: true,
  minCompressChars: 2000,
  // Below ~6k chars, per-image cost dominates savings (break-even territory).
  minReminderChars: 6000,
  minToolResultChars: 6000,
  // system field rejects images (400 system.N.type: Input should be 'text') —
  // images always go into the first user message.
  // 313 cols × 5 px + 8 px pad = 1573 px slab width (under 2000 px ceiling).
  cols: 313,
  maxImagesPerToolResult: 10,
  charsPerToken: 4,
  historyAmortizationHorizon: 1,
  priorWarmTokens: 0,
  priorWarmImageTokens: 0,
  // Multi-col off: single-col slab already holds ~50k chars; extra OCR risk not worth it.
  multiCol: 1,
  reflow: true,
  keepSharp: () => false,
  emitRecoverable: false,
};

// --- per-block break-even check ---
//
// Image token cost is computed from pixel area (Anthropic formula: w×h/750,
// empirically accurate to ~5% on dense PNGs). Constants bias CONSERVATIVE:
// CHARS_PER_TOKEN=4 under-estimates text savings; multi-col cost is linearly
// scaled from single-col + 10% margin. Mispredictions leave money on the
// table; they never generate net-loss images.

/** English ~4 chars per token average (conservative for code/JSON content). */
const CHARS_PER_TOKEN = 4;

/** Empirical cpt for the system-slab path (Opus 4.7 tokenizer, N=391, observed 1.91).
 *  Slab-specific because reminders/tool_results have unknown shape; those stay at 4. */
export const SLAB_CHARS_PER_TOKEN = 2.0;

/** Empirical cpt for the history-collapse path (same Opus 4.7 telemetry as SLAB_CHARS_PER_TOKEN).
 *  History is even denser (tool_use JSON dominates), so 2.0 is doubly conservative. */
export const HISTORY_CHARS_PER_TOKEN = 2.0;

/** Anthropic image-billing formula: `tokens ≈ width × height / 750`.
 *  https://docs.anthropic.com/en/docs/build-with-claude/vision#image-tokens
 *  Accurate to ~5% on dense glyph PNGs (N=14 empirical calibration). The renderer
 *  sizes height to content, so per-block images cost far less than full-canvas. */
const ANTHROPIC_PIXELS_PER_TOKEN = 750;
const IMAGE_COST_SAFETY_MARGIN = 1.10; // 10% conservative bias toward pass-through

/** Width in px of a single-col PNG. Must stay in sync with `renderChunkToPng` (render.ts). */
function singleColWidthPx(cols: number): number {
  return 2 * PAD_X + cols * CELL_W;
}

/** Width in px of a multi-col PNG. Mirrors `multiColWidth()` in render.ts. */
function multiColWidthPx(cols: number, numCols: number): number {
  const n = Math.max(1, numCols | 0);
  if (n === 1) return singleColWidthPx(cols);
  const GUTTER_CELLS = 4; // must match render.ts (not exported)
  return 2 * PAD_X + n * cols * CELL_W + (n - 1) * GUTTER_CELLS * CELL_W;
}

/** Exact image-token cost for `visualRows` at given column/multi-col geometry.
 *  Mirrors the renderer's height math so the gate matches Anthropic billing.
 *  Last image is partial-height; each image cost ∝ pixel area. */
function imageTokensForRows(
  visualRows: number,
  cols: number,
  numCols: number = 1,
  imageCountCap?: number,
  maxCharsPerImage: number = READABLE_CHARS_PER_IMAGE,
): number {
  if (!Number.isFinite(visualRows) || visualRows <= 0) return 0;
  const n = Math.max(1, numCols | 0);
  const widthPx = multiColWidthPx(cols, n);
  const hardLinesPerImg = Math.max(1, Math.floor((MAX_HEIGHT_PX - 2 * PAD_Y) / CELL_H));
  const readableLinesPerCol = Math.max(1, Math.floor(maxCharsPerImage / Math.max(1, cols)));
  const linesPerImg = Math.min(hardLinesPerImg, readableLinesPerCol);
  const rowsPerImage = linesPerImg; // pixel rows per image (height)
  const linesPerImage = linesPerImg * n; // wrapped-text lines per image (n cols side-by-side)
  let imagesNeeded = Math.ceil(visualRows / linesPerImage);
  if (imageCountCap !== undefined && imageCountCap > 0) {
    imagesNeeded = Math.min(imagesNeeded, imageCountCap);
  }
  const fullImages = Math.max(0, imagesNeeded - 1);
  const linesInLast = visualRows - fullImages * linesPerImage;
  // Column-major layout: pixel rows = min(linesInLast, rowsPerImage).
  const rowsInLast = Math.min(Math.max(1, linesInLast), rowsPerImage);
  const fullImageHeight = 2 * PAD_Y + rowsPerImage * CELL_H;
  const lastImageHeight = 2 * PAD_Y + rowsInLast * CELL_H;
  const totalPixels = fullImages * widthPx * fullImageHeight + widthPx * lastImageHeight;
  return Math.ceil((totalPixels / ANTHROPIC_PIXELS_PER_TOKEN) * IMAGE_COST_SAFETY_MARGIN);
}

/** Exact image-token cost for `text`. Uses `countVisualRows` and optionally
 *  `shrinkColsToContent` (default true) so narrow blocks aren't priced at full
 *  canvas width. Pass `shrinkWidth=false` for the system slab (fills full `cols`). */
function imageTokensCost(
  text: string,
  cols: number,
  numCols: number = 1,
  imageCountCap?: number,
  shrinkWidth: boolean = true,
  maxCharsPerImage: number = READABLE_CHARS_PER_IMAGE,
): number {
  const effectiveCols = shrinkWidth ? shrinkColsToContent(text, cols) : cols;
  const rows = countVisualRows(text, effectiveCols);
  return imageTokensForRows(rows, effectiveCols, numCols, imageCountCap, maxCharsPerImage);
}

/** Gate geometry for the single-col dense path (tool_result, reminder, history).
 *  Dense single-col uses DENSE_CONTENT_COLS/DENSE_CONTENT_CHARS_PER_IMAGE;
 *  multi-col uses configured `cols` at READABLE budget. Slab uses its own path. */
function denseGateGeometry(cols: number, numCols: number): { cols: number; maxChars: number } {
  return Math.max(1, numCols | 0) > 1
    ? { cols, maxChars: READABLE_CHARS_PER_IMAGE }
    : { cols: DENSE_CONTENT_COLS, maxChars: DENSE_CONTENT_CHARS_PER_IMAGE };
}

/** Visual rows per image: `floor((MAX_HEIGHT_PX − 2·PAD_Y) / CELL_H)`. Derived
 *  from render.ts constants so break-even math auto-tracks cell geometry changes. */
export const LINES_PER_IMAGE = Math.max(1, Math.floor((MAX_HEIGHT_PX - 2 * PAD_Y) / CELL_H));

export function maxCharsPerImage(cols: number): number {
  return Math.min(cols * LINES_PER_IMAGE, READABLE_CHARS_PER_IMAGE);
}

/** Lossless pre-render whitespace compactor (each `\n` costs ≥1 visual row):
 *  1. Strip trailing whitespace per line (preserves leading indent).
 *  2. Collapse 3+ consecutive newlines to 2. Typically saves 10-25% rows on
 *     markdown/tool-doc slabs, enough to flip borderline gates to profitable. */
export function compactSlabWhitespace(text: string): string {
  if (!text) return text;
  // Single-pass trailing whitespace strip (avoids materializing a split array on ~160 KB slabs).
  let trimmed = '';
  let lineStart = 0;
  for (let i = 0; i <= text.length; i++) {
    if (i === text.length || text.charCodeAt(i) === 10 /* \n */) {
      let end = i;
      while (end > lineStart) {
        const c = text.charCodeAt(end - 1);
        if (c !== 32 && c !== 9) break;
        end--;
      }
      trimmed += text.slice(lineStart, end);
      if (i < text.length) trimmed += '\n';
      lineStart = i + 1;
    }
  }
  // Collapse 3+ newlines → 2 (kills multi-blank dividers; each costs a render row).
  return trimmed.replace(/\n{3,}/g, '\n\n');
}

/** Apply R3 reflow when enabled. Run after `compactSlabWhitespace`, before
 *  the gate (gate/renderer/paging all see the same dense text). Falls back to
 *  input unchanged on sentinel collision. */
function maybeReflow(text: string, enabled: boolean): string {
  if (!enabled) return text;
  return reflow(text) ?? text;
}

/** Decompose the break-even gate into components for telemetry. Returns the
 *  imageTokens, textTokens, and symmetric burn terms the gate uses internally,
 *  or `null` for empty/non-finite input. */
export function evalCompressionProfitability(
  text: string,
  cols: number,
  imageCountCap: number | undefined = undefined,
  numCols: number = 1,
  charsPerToken: number = CHARS_PER_TOKEN,
  priorWarmTokens: number = 0,
  priorWarmImageTokens: number = 0,
  shrinkWidth: boolean = true,
): {
  imageTokens: number;
  textTokens: number;
  burnImageSide: number;
  burnTextSide: number;
  profitable: boolean;
} | null {
  const n = Math.max(1, numCols | 0);
  if (typeof text !== 'string' || text.length === 0) return null;
  const cpt = Number.isFinite(charsPerToken) && charsPerToken > 0
    ? charsPerToken
    : CHARS_PER_TOKEN;
  const imageTokens = imageTokensCost(text, cols, n, imageCountCap, shrinkWidth);
  const textTokens = text.length / cpt;
  const burnImageSide = Number.isFinite(priorWarmTokens) && priorWarmTokens > 0
    ? priorWarmTokens * (CACHE_CREATE_RATE - CACHE_READ_RATE)
    : 0;
  const burnTextSide = Number.isFinite(priorWarmImageTokens) && priorWarmImageTokens > 0
    ? priorWarmImageTokens * (CACHE_CREATE_RATE - CACHE_READ_RATE)
    : 0;
  return {
    imageTokens,
    textTokens,
    burnImageSide,
    burnTextSide,
    profitable: imageTokens + burnImageSide < textTokens + burnTextSide,
  };
}

export function isCompressionProfitable(
  text: string,
  cols: number = DEFAULTS.cols,
  imageCountCap?: number,
  numCols: number = 1,
  charsPerToken: number = CHARS_PER_TOKEN,
  priorWarmTokens: number = 0,
  priorWarmImageTokens: number = 0,
  shrinkWidth: boolean = true,
  maxCharsPerImage: number = READABLE_CHARS_PER_IMAGE,
): boolean {
  const n = Math.max(1, numCols | 0);
  if (typeof text !== 'string' || text.length === 0) return false;
  const cpt = Number.isFinite(charsPerToken) && charsPerToken > 0
    ? charsPerToken
    : CHARS_PER_TOKEN;
  const imageTokensCost_ = imageTokensCost(text, cols, n, imageCountCap, shrinkWidth, maxCharsPerImage);
  const textTokensEquivalent = text.length / cpt;
  // Symmetric burn penalty (anti-flapping): switching modes invalidates the warm
  // cache on whichever side was warm, paying cache_create. Burn is added to the
  // side that would flip — pinning the session in its current mode until
  // per-turn savings exceed the burn cost.
  const burnImageSide = Number.isFinite(priorWarmTokens) && priorWarmTokens > 0
    ? priorWarmTokens * (CACHE_CREATE_RATE - CACHE_READ_RATE)
    : 0;
  const burnTextSide = Number.isFinite(priorWarmImageTokens) && priorWarmImageTokens > 0
    ? priorWarmImageTokens * (CACHE_CREATE_RATE - CACHE_READ_RATE)
    : 0;
  return imageTokensCost_ + burnImageSide < textTokensEquivalent + burnTextSide;
}

/**
 * Horizon-aware variant of `isCompressionProfitable` for history-collapse.
 *
 * Evaluates expected lifetime cost over N turns: worst-case-warm for image
 * (cache_create turn 1, cache_read turns 2..N) vs best-case-warm for text
 * (cache_read all N). Gate condition: I×(CC + CR×(N-1)) < T×CR×N.
 * Examples: N=5 → I < 0.30×T; N=10 → I < 0.47×T.
 * Falls back to cold per-turn gate when `horizon <= 1`. See docs/HISTORY_CACHE_MODEL.md.
 */
export function isCompressionProfitableAmortized(
  text: string,
  cols: number,
  imageCountCap: number | undefined,
  numCols: number,
  charsPerToken: number,
  horizon: number,
  priorWarmTokens: number = 0,
  priorWarmImageTokens: number = 0,
  shrinkWidth: boolean = true,
  maxCharsPerImage: number = READABLE_CHARS_PER_IMAGE,
): boolean {
  if (!Number.isFinite(horizon) || horizon <= 1) {
    return isCompressionProfitable(text, cols, imageCountCap, numCols, charsPerToken, priorWarmTokens, priorWarmImageTokens, shrinkWidth, maxCharsPerImage);
  }
  const N = Math.max(2, Math.floor(horizon));
  const n = Math.max(1, numCols | 0);
  if (typeof text !== 'string' || text.length === 0) return false;
  const cpt = Number.isFinite(charsPerToken) && charsPerToken > 0
    ? charsPerToken
    : CHARS_PER_TOKEN;
  const imageTokens = imageTokensCost(text, cols, n, imageCountCap, shrinkWidth, maxCharsPerImage);
  const textTokens = text.length / cpt;
  // Worst-case-for-image vs best-case-for-text (conservative, on purpose).
  const imageLifetime = imageTokens * (CACHE_CREATE_RATE + CACHE_READ_RATE * (N - 1));
  const textLifetime = textTokens * CACHE_READ_RATE * N;
  // Symmetric burn — see isCompressionProfitable for anti-flapping rationale.
  const burnImageSide = Number.isFinite(priorWarmTokens) && priorWarmTokens > 0
    ? priorWarmTokens * (CACHE_CREATE_RATE - CACHE_READ_RATE)
    : 0;
  const burnTextSide = Number.isFinite(priorWarmImageTokens) && priorWarmImageTokens > 0
    ? priorWarmImageTokens * (CACHE_CREATE_RATE - CACHE_READ_RATE)
    : 0;
  return imageLifetime + burnImageSide < textLifetime + burnTextSide;
}


/** Increment a passthrough-reason counter on `info`. Lazily allocates `passthroughReasons`. */
function bumpPassthrough(
  info: TransformInfo,
  reason: 'below_threshold' | 'not_profitable' | 'kept_sharp',
): void {
  if (!info.passthroughReasons) info.passthroughReasons = {};
  info.passthroughReasons[reason] = (info.passthroughReasons[reason] ?? 0) + 1;
}

/** Invoke `keepSharp` defensively; a throw or non-`true` return means "image as usual". */
function callerKeepsSharp(
  fn: ((block: KeepSharpBlock) => boolean) | undefined,
  block: KeepSharpBlock,
): boolean {
  if (typeof fn !== 'function') return false;
  try {
    return fn(block) === true;
  } catch {
    return false;
  }
}

/** Logical bucket for per-gate-call char attribution. Used by the rolling-cpt
 *  regression to derive per-bucket marginal cpt from production telemetry. */
export type BucketName =
  | 'static_slab'
  | 'reminder'
  | 'tool_result_json'
  | 'tool_result_log'
  | 'tool_result_prose'
  | 'history';

/** Pre-compaction TEXT char totals per bucket. Absent when no bucket fired. */
export type BucketChars = Partial<Record<BucketName, number>>;

/** Attribute `chars` to a compression bucket (called whether gate accepted or rejected). */
function bumpBucket(info: TransformInfo, bucket: BucketName, chars: number): void {
  if (chars <= 0) return;
  if (!info.bucketChars) info.bucketChars = {};
  info.bucketChars[bucket] = (info.bucketChars[bucket] ?? 0) + chars;
}

/** Map `classifyContent` shape to a tool_result bucket name. */
function toolResultBucket(shape: 'structured' | 'log' | 'other'): BucketName {
  if (shape === 'structured') return 'tool_result_json';
  if (shape === 'log') return 'tool_result_log';
  return 'tool_result_prose';
}

/** Parsed contents of Claude Code's <env> + git status blocks. All optional —
 *  fields are only populated if the corresponding line is present. */
export interface EnvFields {
  /** Working directory at the time `claude` was launched. */
  cwd?: string;
  isGitRepo?: boolean;
  /** Current git branch, parsed from <git_status> or a "Branch:" line. */
  gitBranch?: string;
  platform?: string;
  osVersion?: string;
  /** "Today's date" as Claude Code reported it (YYYY-MM-DD). */
  today?: string;
}

export interface TransformInfo {
  compressed: boolean;
  reason?: string;
  origChars: number;
  /** Total source chars image-encoded this request (static slab + reminders + tool_results).
   *  Unlike `origChars` (static slab + tool docs only), reflects what `imageCount` replaced. */
  compressedChars: number;
  imageCount: number;
  imageBytes: number;
  /** Σ width×height across all rendered images. Pairs with upstream token count for
   *  empirical px/token regression: `tokens ≈ α·outgoingTextChars + β·imagePixels`. */
  imagePixels?: number;
  /** Total TEXT chars in the outgoing body (system + messages, excluding image base64).
   *  Denominator for empirical chars-per-token regression on cold-miss events. */
  outgoingTextChars?: number;
  /** Length of the static (cacheable) slab rendered into the image. */
  staticChars: number;
  /** Length of the dynamic (per-turn) slab kept as plain text. */
  dynamicChars: number;
  dynamicBlockCount: number;
  /** Tag-shaped blocks in the static slab not in DYNAMIC_BLOCK_TAGS.
   *  Canary: a new per-turn Claude Code tag would appear here before cache rate collapses. */
  unknownStaticTags?: string[];
  env?: EnvFields;
  /** sha8 of static slab + tool docs (what goes in the image). Repeats across turns → cache hits. */
  systemSha8?: string;
  /** sha8 of the CLAUDE.md section, for bucketing by project when cwd is absent. */
  claudeMdSha8?: string;
  /** sha8 of first user message text (first 4 KiB). Rough thread/session id. */
  firstUserSha8?: string;
  /** Raw bytes of the first rendered image. Dashboard preview only; NOT persisted to JSONL. */
  firstImagePng?: Uint8Array;
  firstImageWidth?: number;
  firstImageHeight?: number;
  /** All rendered PNGs this request. Dashboard only; NOT persisted to JSONL. */
  imagePngs?: Uint8Array[];
  imageDims?: Array<{ width: number; height: number }>;
  /** Source text rendered to images (slab + header), capped at 64 KiB. NOT persisted. */
  imageSourceText?: string;
  reminderImgs?: number;
  toolResultImgs?: number;
  /** Codepoints missing from the atlas (rendered as blank cells). Telemetry for atlas tuning. */
  droppedChars?: number;
  /** Top dropped codepoints by frequency (`U+HHHH` → count), at most 20 entries. */
  droppedCodepointsTop?: Record<string, number>;
  /** Why blocks passed through without compression. Only present when count > 0. */
  passthroughReasons?: { below_threshold?: number; not_profitable?: number; kept_sharp?: number };
  /** Slab gate diagnostics — imageTokens, textTokens, burn terms, and verdict.
   *  Lets hosts measure flap-prevention efficacy and tune amortization horizon. */
  gateEval?: {
    readonly site: 'slab';
    readonly imageTokens: number;
    readonly textTokens: number;
    /** `priorWarmTokens × (CC − CR)` added to image side. */
    readonly burnImageSide: number;
    /** `priorWarmImageTokens × (CC − CR)` added to text side (anti-flapping anchor). */
    readonly burnTextSide: number;
    readonly profitable: boolean;
  };
  /** Pre-compaction TEXT char totals per gate-call bucket. Rolling-cpt regression denominator. */
  bucketChars?: BucketChars;
  /** Chars fed into the history-image renderer. Folded into `bucketChars.history` too. */
  historyTextChars?: number;
  /** Blocks pinned as text by the caller's `keepSharp` predicate this request. */
  keptSharpBlocks?: number;
  /** Imaged live-region blocks with original text + provenance, when `emitRecoverable`. */
  recoverable?: RecoverableBlock[];
  truncatedToolResults?: number;
  omittedChars?: number;
  /** History-collapse: messages collapsed into the synthetic prepended user message. */
  collapsedTurns?: number;
  collapsedChars?: number;
  /** History-collapse images. Also folded into `info.imageCount`. */
  collapsedImages?: number;
  /** sha8 of concatenated history-image base64. Stable across the collapse window →
   *  proves Anthropic's prompt cache can `cache_read` (0.1×) instead of `cache_create`.
   *  A changing hash means cache-key drift is back. Only set when collapse produced images. */
  historyImageSha?: string;
  /** Why the history collapse didn't run (or did). Diagnostic only. */
  historyReason?:
    | 'no_history'
    | 'prefix_too_short'
    | 'no_closed_prefix'
    | 'not_profitable'
    | 'render_empty'
    | 'collapsed';
  /** Token count of the pre-compression body from /v1/messages/count_tokens (free).
   *  Absent when probe failed — event excluded from savings rollup. */
  baselineTokens?: number;
  /** Token count of the pre-compression body truncated at the last cache_control marker.
   *  Absent when the original body has no cache_control markers (cacheable=0 exactly). */
  baselineCacheableTokens?: number;
  /** 'ok': both probes resolved. 'partial': full-body resolved but cacheable-prefix
   *  didn't (exclude from rollup — cacheable=0 fallback is dishonest). 'failed': no
   *  baseline. undefined: no probe attempted. */
  baselineProbeStatus?: 'ok' | 'partial' | 'failed';
}

// --- helpers ---------------------------------------------------------------

/** Extract (text, remainder) from a system field that may be string or block list. */
function extractSystemText(sys: SystemField | undefined): { text: string; kept: SystemField } {
  if (sys == null) return { text: '', kept: [] };
  if (typeof sys === 'string') return { text: sys, kept: '' };
  const textParts: string[] = [];
  const kept: SystemField = [];
  for (const block of sys) {
    if (block && typeof block === 'object' && block.type === 'text') {
      textParts.push(block.text);
    } else {
      kept.push(block);
    }
  }
  return { text: textParts.join('\n\n'), kept };
}

function lastStaticSystemCacheControl(sys: SystemField | undefined): TextBlock['cache_control'] | undefined {
  if (!Array.isArray(sys)) return undefined;
  let cacheControl: TextBlock['cache_control'] | undefined;
  for (const block of sys) {
    if (!block || block.type !== 'text' || block.cache_control === undefined) continue;
    const { body } = stripBillingLine(block.text);
    if (splitStaticDynamic(body).staticText.length > 0) {
      cacheControl = block.cache_control;
    }
  }
  return cacheControl;
}

// Per-turn dynamic blocks injected by Claude Code. These drift turn-to-turn and
// must not be baked into the cached image. Split out so only the stable static
// slab (CLAUDE.md + tool docs) carries cache_control.
const DYNAMIC_BLOCK_TAGS = [
  'env',
  'context',
  'git_status',
  'directoryStructure',
  'system-reminder',
] as const;

// Known-static tags in the slab (part of Claude Code's built-in prompt, not per-turn).
// Listed here so the canary in splitStaticDynamic doesn't false-fire on them.
// Add a tag only after confirming it doesn't rotate per turn.
const KNOWN_STATIC_TAGS = ['types'] as const;

function splitStaticDynamic(text: string): {
  staticText: string;
  dynamicText: string;
  blockCount: number;
  unknownTags: string[];
} {
  if (!text)
    return { staticText: '', dynamicText: '', blockCount: 0, unknownTags: [] };
  const pattern = new RegExp(
    `<(${DYNAMIC_BLOCK_TAGS.join('|')})(\\s[^>]*)?>[\\s\\S]*?</\\1>`,
    'g',
  );
  const dynamicParts: string[] = [];
  let staticBuf = '';
  let cursor = 0;
  let m: RegExpExecArray | null;
  while ((m = pattern.exec(text)) !== null) {
    staticBuf += text.slice(cursor, m.index);
    dynamicParts.push(m[0]);
    cursor = m.index + m[0].length;
  }
  staticBuf += text.slice(cursor);

  // Sniff for unknown tag-shaped blocks in the static slab. A new per-turn
  // Claude Code tag would silently bake into the image and collapse cache rate;
  // surfacing the tag name lets us detect it within hours of a release.
  const known = new Set<string>(DYNAMIC_BLOCK_TAGS);
  const knownStatic = new Set<string>(KNOWN_STATIC_TAGS);
  const sniffer = /<([a-zA-Z][a-zA-Z0-9_-]*)(?:\s[^>]*)?>[\s\S]*?<\/\1>/g;
  const unknown = new Set<string>();
  let s: RegExpExecArray | null;
  while ((s = sniffer.exec(staticBuf)) !== null) {
    const tag = s[1]!;
    if (!known.has(tag) && !knownStatic.has(tag) && tag.length <= 64)
      unknown.add(tag);
  }

  return {
    // Collapse the run of blank lines left behind by removed blocks.
    staticText: staticBuf.replace(/\n{3,}/g, '\n\n').trim(),
    dynamicText: dynamicParts.join('\n\n'),
    blockCount: dynamicParts.length,
    unknownTags: [...unknown],
  };
}

/** sha256[0..8] hex via Web Crypto (works in Node 18+ and Workers). 32-bit collision-safe. */
export async function sha8(text: string): Promise<string> {
  const buf = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest('SHA-256', buf);
  const bytes = new Uint8Array(digest);
  let hex = '';
  for (let i = 0; i < 4; i++) hex += bytes[i]!.toString(16).padStart(2, '0');
  return hex;
}

/** Record a recovery entry when `emitRecoverable` is on. No-op (no hash cost) when off. */
async function recordRecoverable(
  info: TransformInfo,
  emit: boolean,
  entry: { kind: RecoverableBlock['kind']; toolUseId?: string; text: string; imageCount: number },
): Promise<void> {
  if (!emit) return;
  const id = 'rec_' + (await sha8(`${entry.kind}\u0000${entry.toolUseId ?? ''}\u0000${entry.text}`));
  (info.recoverable ??= []).push({
    id,
    kind: entry.kind,
    ...(entry.toolUseId !== undefined ? { toolUseId: entry.toolUseId } : {}),
    text: entry.text,
    imageCount: entry.imageCount,
  });
}

/** Hash the concatenated base64 of every image block on `messages[0]` (the synthetic
 *  history message). Stable across the quantized collapse window → proves Anthropic
 *  can cache_read the history prefix. Returns undefined if no images on messages[0]. */
async function historyImageSha8(
  messages: Message[],
): Promise<string | undefined> {
  const synthetic = messages[0];
  if (!synthetic || !Array.isArray(synthetic.content)) return undefined;
  let concat = '';
  for (const blk of synthetic.content) {
    if (blk.type === 'image') concat += blk.source.data;
  }
  return concat ? sha8(concat) : undefined;
}

/** Best-effort extraction of the CLAUDE.md slab from a system text (heuristic).
 *  Returns empty string if nothing CLAUDE.md-shaped is detected. */
export function extractClaudeMdSlab(staticText: string): string {
  if (!staticText) return '';
  // Headings Claude Code uses around CLAUDE.md content.
  const startPatterns = [
    /^\s*#+\s*Claude\s+Code\s+Rules\s*$/im,
    /^\s*#+\s*CLAUDE\.md\s*$/im,
    /^\s*Claude\s+Code\s+Rules:?\s*$/im,
  ];
  let startIdx = -1;
  for (const p of startPatterns) {
    const m = p.exec(staticText);
    if (m && (startIdx === -1 || m.index < startIdx)) startIdx = m.index;
  }
  if (startIdx === -1) return '';
  // End at the next top-level heading or EOF.
  const tail = staticText.slice(startIdx);
  const endMatch = /\n#\s+\S/.exec(tail.slice(1));
  const end = endMatch ? endMatch.index + 1 : tail.length;
  return tail.slice(0, end).trim();
}

/** First user message text, capped at 4 KiB (stable thread id; hashing large pastes is wasteful). */
export function firstUserText(req: MessagesRequest): string {
  const msgs = req.messages ?? [];
  for (const m of msgs) {
    if (m.role !== 'user') continue;
    if (typeof m.content === 'string') return m.content.slice(0, 4096);
    if (Array.isArray(m.content)) {
      for (const block of m.content) {
        if (block && (block as any).type === 'text' && typeof (block as any).text === 'string') {
          return ((block as any).text as string).slice(0, 4096);
        }
      }
    }
    // First user message found but unreadable — return empty rather than fall through to next.
    return '';
  }
  return '';
}

/** Parse structured fields from the dynamic slab for telemetry. Read-only. */
export function extractEnvFields(dynamicText: string): EnvFields {
  const out: EnvFields = {};
  if (!dynamicText) return out;

  const envMatch = /<env>([\s\S]*?)<\/env>/i.exec(dynamicText);
  if (envMatch) {
    const body = envMatch[1]!;
    const cwd = /(?:^|\n)\s*Working directory:\s*(.+?)\s*(?:\n|$)/i.exec(body);
    if (cwd) out.cwd = cwd[1]!.trim();
    const gitRepo = /(?:^|\n)\s*Is directory a git repo:\s*(Yes|No)\b/i.exec(body);
    if (gitRepo) out.isGitRepo = gitRepo[1]!.toLowerCase() === 'yes';
    const platform = /(?:^|\n)\s*Platform:\s*(.+?)\s*(?:\n|$)/i.exec(body);
    if (platform) out.platform = platform[1]!.trim();
    const osVer = /(?:^|\n)\s*OS Version:\s*(.+?)\s*(?:\n|$)/i.exec(body);
    if (osVer) out.osVersion = osVer[1]!.trim();
    const today = /(?:^|\n)\s*Today'?s date:\s*(.+?)\s*(?:\n|$)/i.exec(body);
    if (today) out.today = today[1]!.trim();
  }

  // Branch may be in <git_status>, <context name="git">, or a bare "Branch:" / "On branch" line.
  const branch =
    /(?:^|\n)\s*(?:On branch|Branch:)\s*([^\s\n]+)/i.exec(dynamicText) ??
    /(?:^|\n)\s*Current branch:\s*([^\s\n]+)/i.exec(dynamicText);
  if (branch) out.gitBranch = branch[1]!.trim();

  return out;
}

/** Strip the per-turn `x-anthropic-billing-header:` line (changes every turn;
 *  must not be baked into the image). Returned as `kept` for the system tail. */
function stripBillingLine(text: string): { kept: string | null; body: string } {
  const nl = text.indexOf('\n');
  const first = nl === -1 ? text : text.slice(0, nl);
  if (first.startsWith('x-anthropic-billing-header:')) {
    return { kept: first, body: nl === -1 ? '' : text.slice(nl + 1) };
  }
  return { kept: null, body: text };
}

/** Max recursion depth for schema stripping. 20 handles realistic DSL/query schemas;
 *  deeper nodes are left untouched rather than corrupted. */
const SCHEMA_STRIP_MAX_DEPTH = 20;

/** Metadata keys that add tokens but no validation; the image carries them for the model. */
const SCHEMA_STRIP_KEYS = new Set([
  'description',
  'title',
  'examples',
  'default',
  '$schema',
  '$id',
  '$comment',
]);

/** JSON Schema composition keys (values are arrays of subschemas). */
const SCHEMA_COMPOSITION_KEYS = new Set(['oneOf', 'anyOf', 'allOf']);

/** JSON Schema keys whose values are named-subschema objects. */
const SCHEMA_NAMED_SUBSCHEMA_KEYS = new Set([
  'properties',
  'patternProperties',
  'definitions',
  '$defs',
]);

/** JSON Schema keys whose value is a single subschema. */
const SCHEMA_SINGLE_SUBSCHEMA_KEYS = new Set([
  'items',
  'additionalProperties',
  'not',
  'contains',
  'propertyNames',
  'unevaluatedItems',
  'unevaluatedProperties',
  'if',
  'then',
  'else',
]);

/** JSON Schema keys that are primitives or opaque arrays — pass through verbatim. */
const SCHEMA_VERBATIM_KEYS = new Set([
  'required',
  'enum',
  'const',
  'type',          // string or array of strings
  '$ref',          // we don't resolve refs but we mustn't drop them
  'minimum',
  'maximum',
  'exclusiveMinimum',
  'exclusiveMaximum',
  'minLength',
  'maxLength',
  'minItems',
  'maxItems',
  'minProperties',
  'maxProperties',
  'multipleOf',
  'uniqueItems',
  'pattern',
]);

/** Real `format` tokens (date-time, uri, email…) are short; anything longer is a description. */
const FORMAT_MAX_LEN = 32;

/** Strip long-form metadata from a JSON Schema node, preserving structural keys
 *  Anthropic's tool-use validator needs. Strips: description, title, examples, default,
 *  $schema, $id, $comment, long format. Recurses into properties/oneOf/anyOf/allOf/items
 *  etc. Returns a fresh object — never mutates the input. */
function stripSchemaDescriptions(node: unknown, depth: number): unknown {
  if (depth > SCHEMA_STRIP_MAX_DEPTH) return node; // leave pathological depth untouched
  if (Array.isArray(node)) return node; // subschema arrays handled by parent
  if (!node || typeof node !== 'object') return node;

  const obj = node as Record<string, unknown>;
  const out: Record<string, unknown> = {};

  for (const [k, v] of Object.entries(obj)) {
    if (SCHEMA_STRIP_KEYS.has(k)) continue;

    if (k === 'format' && typeof v === 'string' && v.length > FORMAT_MAX_LEN) {
      continue; // long format = description in disguise
    }

    if (SCHEMA_VERBATIM_KEYS.has(k)) {
      out[k] = v;
      continue;
    }

    if (
      SCHEMA_NAMED_SUBSCHEMA_KEYS.has(k) &&
      v &&
      typeof v === 'object' &&
      !Array.isArray(v)
    ) {
      const nested: Record<string, unknown> = {};
      for (const [pk, pv] of Object.entries(v as Record<string, unknown>)) {
        nested[pk] = stripSchemaDescriptions(pv, depth + 1);
      }
      out[k] = nested;
      continue;
    }

    if (SCHEMA_COMPOSITION_KEYS.has(k) && Array.isArray(v)) {
      out[k] = v.map((sub) => stripSchemaDescriptions(sub, depth + 1));
      continue;
    }

    if (SCHEMA_SINGLE_SUBSCHEMA_KEYS.has(k)) {
      // additionalProperties may be a boolean — pass through untouched.
      if (typeof v === 'boolean') {
        out[k] = v;
      } else {
        out[k] = stripSchemaDescriptions(v, depth + 1);
      }
      continue;
    }

    // Unknown key — recurse into nested objects so vendor-extension descriptions get stripped.
    if (v && typeof v === 'object') {
      out[k] = stripSchemaDescriptions(v, depth + 1);
    } else {
      out[k] = v;
    }
  }
  return out;
}

/** Keys that give Anthropic's validator something to bind against. A stripped schema
 *  with none of these gets the bare `{type:'object'}` stub + schema_no_properties advisory. */
const SCHEMA_STRUCTURAL_KEYS = [
  'properties',
  'patternProperties',
  'oneOf',
  'anyOf',
  'allOf',
  'items',
  '$ref',
  'enum',
  'const',
];

function schemaHasStructure(schema: Record<string, unknown>): boolean {
  for (const k of SCHEMA_STRUCTURAL_KEYS) {
    if (k in schema) return true;
  }
  return false;
}

/** Build the "## Tool: name\n<desc>\n<schema>" block for one tool. Schema is serialized
 *  compact (no whitespace) — pretty-print wastes 70%+ of horizontal space per key. */
function renderToolDoc(t: ToolDef, includeSchema: boolean): string {
  const parts: string[] = [`## Tool: ${t.name ?? '?'}`];
  if (t.description) parts.push(t.description);
  if (includeSchema && t.input_schema !== undefined) {
    parts.push('```json\n' + JSON.stringify(t.input_schema) + '\n```');
  }
  return parts.join('\n');
}

function makeImageBlock(pngB64: string, _ephemeral = false): ImageBlock {
  // pxpipe never adds its own cache_control — only moves existing caller markers
  // across the text→image flip. `_ephemeral` is preserved for call-site compat.
  return {
    type: 'image',
    source: { type: 'base64', media_type: 'image/png', data: pngB64 },
  };
}

// --- paging / truncation ---------------------------------------------------
// Anthropic caps requests at 100 images. Huge tool_results (find trees,
// log dumps) are truncated with a paging marker before render.

/** Visual rows a single input line will consume after soft-wrap at `cols`. */
function lineRows(line: string, cols: number): number {
  return Math.max(1, Math.ceil(line.length / cols));
}

/** Visual row count after soft-wrap at `cols`. Both `\n` and the ↵ sentinel
 *  end a row; ↵ occupies a cell on the line it terminates. */
function countVisualRows(text: string, cols: number): number {
  let rows = 0;
  let lineStart = 0;
  const len = text.length;
  for (let i = 0; i <= len; i++) {
    const cc = i < len ? text.charCodeAt(i) : -1;
    const isSentinel = cc === 0x21b5 /* ↵ */;
    if (i === len || cc === 10 /* \n */ || isSentinel) {
      // ↵ renders as a glyph on the line it ends — count it in the length.
      const lineLen = (isSentinel ? i + 1 : i) - lineStart;
      rows += Math.max(1, Math.ceil(lineLen / cols));
      lineStart = i + 1;
    }
  }
  return rows;
}

/** Estimate how many images `text` will render to at the given column width.
 *  Counts soft-wrapped visual rows, which is what render.ts actually budgets
 *  against. Exported for tests + the paging gate.
 *
 *  `numCols` (default 1) packs that many text columns side-by-side per
 *  image — must match the `multiCol` setting wired through to the renderer
 *  for the math to predict the actual image count. */
export function estimateImageCount(
  textOrLen: string | number,
  cols: number,
  numCols: number = 1,
  maxCharsPerImage: number = READABLE_CHARS_PER_IMAGE,
): number {
  const n = Math.max(1, numCols | 0);
  const readableLinesPerCol = Math.max(1, Math.floor(maxCharsPerImage / Math.max(1, cols)));
  const linesPerImage = Math.min(LINES_PER_IMAGE, readableLinesPerCol) * n;
  const charBudget = Math.max(1, maxCharsPerImage * n);
  if (typeof textOrLen === 'number') {
    // Back-compat shim — numeric arg gets the looser chars-based estimate.
    return Math.max(1, Math.ceil(textOrLen / charBudget));
  }
  const rows = countVisualRows(textOrLen, cols);
  return Math.max(
    1,
    Math.ceil(rows / linesPerImage),
    Math.ceil(textOrLen.length / charBudget),
  );
}

/** Classify content so we can pick a truncation strategy. Cheap heuristics on
 *  the first ~4 KiB. Returns:
 *    - `'structured'`: JSON/YAML/diff markers at the top. Truncate tail.
 *    - `'log'`: ≥30% of lines start with a log level or timestamp. Truncate middle.
 *    - `'other'`: prose, file dumps, etc. Truncate middle.
 *  Exported for tests. */
export function classifyContent(text: string): 'structured' | 'log' | 'other' {
  const head = text.slice(0, 4096);
  const trimmed = head.trimStart();
  if (trimmed.startsWith('{') && /^\{\s*("|\})/.test(trimmed)) return 'structured';
  if (trimmed.startsWith('[') && /^\[\s*("|\{|\[|-?\d|true\b|false\b|null\b|\])/.test(trimmed))
    return 'structured';
  if (trimmed.startsWith('---\n') || trimmed.startsWith('---\r\n')) return 'structured';
  if (trimmed.startsWith('diff --git ') || /^---\s+\S/.test(trimmed)) return 'structured';
  const lines = head.split('\n').slice(0, 40).filter((l) => l.length > 0);
  if (lines.length < 4) return 'other';
  const LOG_LINE =
    /^(\[?(DEBUG|INFO|WARN|WARNING|ERROR|TRACE|FATAL)\]?\b|\d{4}-\d{2}-\d{2}[T ]?|\d{2}:\d{2}:\d{2}\b)/;
  let logHits = 0;
  for (const line of lines) if (LOG_LINE.test(line)) logHits++;
  if (logHits / lines.length >= 0.3) return 'log';
  return 'other';
}

/** Build the paging marker text. The model sees this verbatim INSIDE the
 *  rendered image so it can reason about what was elided. */
function buildPagingMarker(args: {
  originalChars: number;
  originalLines: number;
  originalEstImages: number;
  shownHeadLines: number;
  shownTailLines: number;
  omittedLines: number;
  omittedChars: number;
}): string {
  const tailNote =
    args.shownTailLines > 0
      ? ` Showing first ${args.shownHeadLines} lines and last ${args.shownTailLines} lines.`
      : ` Showing first ${args.shownHeadLines} lines (tail elided).`;
  return (
    `\n\n[ pxpipe paging: omitted ${args.omittedLines.toLocaleString('en-US')} lines ` +
    `(${args.omittedChars.toLocaleString('en-US')} chars) of content here. ` +
    `Original length: ${args.originalChars.toLocaleString('en-US')} chars ` +
    `(${args.originalLines.toLocaleString('en-US')} lines, ~${args.originalEstImages} images).` +
    `${tailNote} ]\n\n`
  );
}

/** Truncate `text` so it renders to roughly `maxImages` images at the given
 *  `cols`. Picks head/tail split based on `classifyContent`. Budget measured
 *  in visual rows (what render.ts actually slices on). Returns the truncated
 *  text (with paging marker embedded) and the count of chars omitted. If
 *  `text` already fits, returns unchanged with `omittedChars: 0`. Exported
 *  for tests. */
export function truncateForBudget(
  text: string,
  maxImages: number,
  cols: number,
  numCols: number = 1,
  maxCharsPerImage: number = DENSE_CONTENT_CHARS_PER_IMAGE,
): { text: string; omittedChars: number; truncated: boolean } {
  const n = Math.max(1, numCols | 0);
  const estImages = estimateImageCount(text, cols, n, maxCharsPerImage);
  if (estImages <= maxImages) return { text, omittedChars: 0, truncated: false };
  const readableLinesPerCol = Math.max(1, Math.floor(maxCharsPerImage / Math.max(1, cols)));
  const totalRowBudget = Math.max(8, maxImages * Math.min(LINES_PER_IMAGE, readableLinesPerCol) * n - 6);
  const totalCharBudget = Math.max(128, maxImages * maxCharsPerImage * n - 512);
  const shape = classifyContent(text);
  // Reflowed text uses NL_SENTINEL (↵ U+21B5) as line separator instead of \n.
  // Split on whichever delimiter the text uses so we can truncate at logical
  // line boundaries rather than treating the entire reflowed blob as one line.
  const nlChar = text.indexOf('\n') >= 0 ? '\n' : NL_SENTINEL;
  const lines = text.split(nlChar);
  const originalLines = lines.length;
  const originalChars = text.length;

  if (shape === 'structured') {
    let rows = 0;
    let chars = 0;
    let cut = 0;
    for (let i = 0; i < lines.length; i++) {
      const r = lineRows(lines[i]!, cols);
      const c = lines[i]!.length + (i > 0 ? 1 : 0);
      if (rows + r > totalRowBudget || chars + c > totalCharBudget) break;
      rows += r;
      chars += c;
      cut = i + 1;
    }
    if (cut === 0) cut = 1;
    const head = lines.slice(0, cut).join(nlChar);
    const omitted = originalChars - head.length;
    return {
      text:
        head +
        buildPagingMarker({
          originalChars,
          originalLines,
          originalEstImages: estImages,
          shownHeadLines: cut,
          shownTailLines: 0,
          omittedLines: originalLines - cut,
          omittedChars: omitted,
        }),
      omittedChars: omitted,
      truncated: true,
    };
  }

  // log / other: 60% head, 40% tail.
  const headRowBudget = Math.floor(totalRowBudget * 0.6);
  const tailRowBudget = totalRowBudget - headRowBudget;
  const headCharBudget = Math.floor(totalCharBudget * 0.6);
  const tailCharBudget = totalCharBudget - headCharBudget;
  let headRows = 0;
  let headChars = 0;
  let headCut = 0;
  for (let i = 0; i < lines.length; i++) {
    const r = lineRows(lines[i]!, cols);
    const c = lines[i]!.length + (i > 0 ? 1 : 0);
    if (headRows + r > headRowBudget || headChars + c > headCharBudget) break;
    headRows += r;
    headChars += c;
    headCut = i + 1;
  }
  if (headCut === 0) headCut = 1;
  let tailRows = 0;
  let tailChars = 0;
  let tailStart = lines.length;
  for (let i = lines.length - 1; i >= headCut; i--) {
    const r = lineRows(lines[i]!, cols);
    const c = lines[i]!.length + (i < lines.length - 1 ? 1 : 0);
    if (tailRows + r > tailRowBudget || tailChars + c > tailCharBudget) break;
    tailRows += r;
    tailChars += c;
    tailStart = i;
  }
  if (tailStart <= headCut || tailStart >= lines.length) {
    const head = lines.slice(0, headCut).join(nlChar);
    const omitted = originalChars - head.length;
    return {
      text:
        head +
        buildPagingMarker({
          originalChars,
          originalLines,
          originalEstImages: estImages,
          shownHeadLines: headCut,
          shownTailLines: 0,
          omittedLines: originalLines - headCut,
          omittedChars: omitted,
        }),
      omittedChars: omitted,
      truncated: true,
    };
  }
  const headText = lines.slice(0, headCut).join(nlChar);
  const tailText = lines.slice(tailStart).join(nlChar);
  const shownChars = headText.length + tailText.length;
  const omitted = originalChars - shownChars;
  return {
    text:
      headText +
      buildPagingMarker({
        originalChars,
        originalLines,
        originalEstImages: estImages,
        shownHeadLines: headCut,
        shownTailLines: lines.length - tailStart,
        omittedLines: originalLines - headCut - (lines.length - tailStart),
        omittedChars: omitted,
      }) +
      tailText,
    omittedChars: omitted,
    truncated: true,
  };
}

async function textToImageBlocks(
  text: string,
  cols: number,
  numCols: number = 1,
  /** Shrink canvas to the longest wrapped line. `false` for the slab path
   *  (fills full `cols` for multi-col packing). Default `true`. */
  shrinkWidth: boolean = true,
): Promise<{
  blocks: ImageBlock[];
  /** Raw PNG bytes parallel to `blocks` (avoids re-decoding base64 for dashboard). */
  pngs: Uint8Array[];
  /** Pixel dimensions parallel to `pngs`. */
  dims: Array<{ width: number; height: number }>;
  droppedChars: number;
  droppedCodepoints: Map<number, number>;
  /** Σ width×height — caller accumulates into `info.imagePixels` for px/token regression. */
  pixels: number;
}> {
  // Shrink before the numCols branch so gate and renderer see the same canvas width.
  // If shrinkage drops below the full width, stay single-col (avoid wasting a divider column).
  const effectiveCols = shrinkWidth ? shrinkColsToContent(text, cols) : cols;
  const effectiveNumCols = effectiveCols < cols ? 1 : numCols;
  const imgs =
    effectiveNumCols > 1
      ? await renderTextToPngsMultiCol(text, effectiveCols, effectiveNumCols)
      : await renderTextToPngsWithCharLimit(text, DENSE_CONTENT_COLS, DENSE_CONTENT_CHARS_PER_IMAGE, DENSE_RENDER_STYLE);
  let droppedChars = 0;
  let pixels = 0;
  const droppedCodepoints = new Map<number, number>();
  const blocks: ImageBlock[] = [];
  for (const img of imgs) {
    blocks.push(makeImageBlock(bytesToBase64(img.png), false));
    droppedChars += img.droppedChars;
    pixels += img.width * img.height;
    for (const [cp, n] of img.droppedCodepoints) {
      droppedCodepoints.set(cp, (droppedCodepoints.get(cp) ?? 0) + n);
    }
  }
  return {
    blocks,
    pngs: imgs.map((i) => i.png),
    dims: imgs.map((i) => ({ width: i.width, height: i.height })),
    droppedChars,
    droppedCodepoints,
    pixels,
  };
}

/** Best-effort byte-count of an image block's PNG payload (decoded from b64).
 *  Used only for the imageBytes telemetry; an exact value isn't worth a
 *  second base64 round-trip. */
function approxBlockBytes(blk: ImageBlock): number {
  const b64 = blk.source.data;
  // base64 → bytes: every 4 chars decode to 3 bytes, minus padding.
  const pad = b64.endsWith('==') ? 2 : b64.endsWith('=') ? 1 : 0;
  return Math.floor((b64.length * 3) / 4) - pad;
}

// --- main transform --------------------------------------------------------


/**
 * Run history-image compression on `req.messages` and finalize the body.
 * Called from both the main path AND early-exit paths (below_min_chars,
 * not_profitable) — history collapse must run even when the slab skips.
 * Tolerant to missing/short message arrays (collapseHistory short-circuits). */
async function runHistoryCollapseAndFinalize(
  req: MessagesRequest,
  info: TransformInfo,
  o: Required<TransformOptions>,
  opts: TransformOptions,
  droppedCodepoints: Map<number, number>,
): Promise<{ body: Uint8Array; info: TransformInfo; collapsed: boolean }> {
  let collapsedFlag = false;
  if (Array.isArray(req.messages) && req.messages.length > 0) {
    const historyCpt = opts.charsPerToken !== undefined
      ? o.charsPerToken
      : HISTORY_CHARS_PER_TOKEN;
    const horizon = Math.max(1, Math.floor(o.historyAmortizationHorizon));
    // Pass the symmetric warm-cache burn through to the history-collapse
    // gate as well. The slab gate alone got the symmetric treatment, which
    // let the history gate flip a session out of image mode even when
    // symmetric burn would have kept the slab gate in. Production data
    // 2026-05-23 showed three-turn sessions paying cache_create every
    // turn because the history gate ignored priorWarmImageTokens.
    const historyProfitable = (text: string, cols: number): boolean => {
      // History always renders single-col at the dense 384-col / 240-row page
      // (history.ts → renderTextToPngsWithCharLimit with DENSE_CONTENT_COLS /
      // DENSE_CONTENT_CHARS_PER_IMAGE), so gate at THAT geometry, not o.cols.
      const g = denseGateGeometry(cols, 1);
      return isCompressionProfitableAmortized(
        text, g.cols, undefined, 1, historyCpt, horizon,
        o.priorWarmTokens, o.priorWarmImageTokens, true, g.maxChars,
      );
    };
    // No protectedPrefix here: this path runs only when the slab did NOT image
    // (it stays as text in req.system), so there is no slab message to shield —
    // collapsing from the head is correct.
    const { messages: newMessages, info: histInfo } = await collapseHistory(
      req.messages,
      historyProfitable,
      { cols: o.cols, protectedPrefix: 0 },
    );
    if (histInfo.collapsedTurns > 0) {
      req.messages = newMessages;
      info.collapsedTurns = histInfo.collapsedTurns;
      info.collapsedChars = histInfo.collapsedChars;
      info.collapsedImages = histInfo.collapsedImages;
      info.imageCount += histInfo.collapsedImages;
      info.imageBytes += histInfo.collapsedImageBytes;
      info.imagePixels = (info.imagePixels ?? 0) + histInfo.collapsedImagePixels;
      info.droppedChars = (info.droppedChars ?? 0) + histInfo.droppedChars;
      for (const [cp, n] of histInfo.droppedCodepoints) {
        droppedCodepoints.set(cp, (droppedCodepoints.get(cp) ?? 0) + n);
      }
      info.historyReason = 'collapsed';
      info.historyTextChars = histInfo.collapsedChars;
      info.historyImageSha = await historyImageSha8(newMessages);
      bumpBucket(info, 'history', histInfo.collapsedChars);
      collapsedFlag = true;
    } else if (histInfo.reason) {
      info.historyReason = histInfo.reason;
    }
  }
  info.outgoingTextChars = countOutgoingTextChars(req);
  const outBody = new TextEncoder().encode(JSON.stringify(req));
  return { body: outBody, info, collapsed: collapsedFlag };
}

/**
 * Rewrite a Messages API request body. Returns the new body (still JSON
 * bytes) plus diagnostic info. On any error, returns the original bytes
 * unchanged.
 */
export async function transformRequest(
  body: Uint8Array,
  opts: TransformOptions = {},
): Promise<{ body: Uint8Array; info: TransformInfo }> {
  // Merge caller opts over DEFAULTS, but treat explicit `undefined` as "not
  // provided" so it falls through to the default. Without this, a caller that
  // passes `{ minToolResultChars: undefined }` (common when forwarding partial
  // options from upstream — e.g. ocproxy's handler) would silently disable the
  // tool_result text-passthrough gate and route everything through the
  // renderer.
  const merged: TransformOptions = { ...DEFAULTS, ...opts };
  for (const k of Object.keys(merged) as (keyof TransformOptions)[]) {
    if (merged[k] === undefined) {
      (merged as Record<string, unknown>)[k] = (DEFAULTS as Record<string, unknown>)[k];
    }
  }
  const o: Required<TransformOptions> = merged as Required<TransformOptions>;
  const info: TransformInfo = {
    compressed: false,
    origChars: 0,
    compressedChars: 0,
    imageCount: 0,
    imageBytes: 0,
    staticChars: 0,
    dynamicChars: 0,
    dynamicBlockCount: 0,
    droppedChars: 0,
  };
  // Per-request codepoint drop histogram. Merged from every render call
  // (static slab + reminder + tool_result compressions). Serialized to
  // `info.droppedCodepointsTop` at the end of transformRequest IF non-empty.
  const droppedCodepoints = new Map<number, number>();

  if (!o.compress) {
    info.reason = 'compress=false';
    return { body, info };
  }

  let req: MessagesRequest;
  try {
    req = JSON.parse(new TextDecoder().decode(body));
  } catch (e) {
    info.reason = `parse_error: ${(e as Error).message}`;
    return { body, info };
  }

  // 1. Pull system text out. Split into:
  //    - billingLine: Claude Code's per-turn random header (must NOT be cached).
  //    - dynamicText: <env>/<context>/... blocks (per-turn, kept as text).
  //    - staticText: everything else (cacheable, goes into the image).
  const systemStaticCacheControl = lastStaticSystemCacheControl(req.system);
  const { text: rawSysText, kept: sysRemainder } = extractSystemText(req.system);
  const { kept: billingLine, body: sysBody } = stripBillingLine(rawSysText);
  const {
    staticText,
    dynamicText,
    blockCount: dynBlocks,
    unknownTags,
  } = splitStaticDynamic(sysBody);
  info.staticChars = staticText.length;
  info.dynamicChars = dynamicText.length;
  info.dynamicBlockCount = dynBlocks;
  if (unknownTags.length > 0) info.unknownStaticTags = unknownTags;
  // Parse env fields out of the dynamic slab — telemetry only, never mutates.
  const env = extractEnvFields(dynamicText);
  if (Object.keys(env).length > 0) info.env = env;

  // Privacy-safe fingerprints that don't depend on tool docs (computed
  // here so they're available even if we below_min_chars out below).
  // systemSha8 is set later, after we know the combined image-bound text.
  const claudeMdSlab = extractClaudeMdSlab(staticText);
  const firstUser = firstUserText(req);
  const [claudeMdSha, firstUserSha] = await Promise.all([
    claudeMdSlab ? sha8(claudeMdSlab) : Promise.resolve(undefined),
    firstUser ? sha8(firstUser) : Promise.resolve(undefined),
  ]);
  if (claudeMdSha) info.claudeMdSha8 = claudeMdSha;
  if (firstUserSha) info.firstUserSha8 = firstUserSha;

  // 2. Optionally fold tool docs into the same image, stubbing originals.
  let toolDocsText = '';
  let toolsRewritten: ToolDef[] | undefined;
  if (o.compressTools && Array.isArray(req.tools) && req.tools.length > 0) {
    const docs: string[] = [];
    let sawSchemaNoProps = false;
    toolsRewritten = req.tools.map((t) => {
      docs.push(renderToolDoc(t, o.compressSchemas));
      // Preserve schema structure (type/properties/required/enum/items) for Anthropic's
      // tool-use validator. Bare {type:'object'} caused 400s on non-interactive turns
      // where Anthropic deep-validates with no prior tool_use history to short-circuit.
      let stubSchema: unknown | undefined;
      if (o.compressSchemas) {
        if (t.input_schema && typeof t.input_schema === 'object') {
          const stripped = stripSchemaDescriptions(
            t.input_schema,
            0,
          ) as Record<string, unknown> | null;
          if (!stripped || typeof stripped !== 'object') {
            // Should not happen for object input, but be defensive.
            stubSchema = { type: 'object' };
            sawSchemaNoProps = true;
          } else if (schemaHasStructure(stripped)) {
            stubSchema = stripped;
          } else {
            // No structural keys → no parameter contract. Ship bare stub and flag it.
            stubSchema = { type: 'object' };
            sawSchemaNoProps = true;
          }
        }
        // input_schema missing entirely → leave field off; don't invent one.
      }
      return {
        ...t,
        description: 'ⓘ See image.',
        ...(stubSchema !== undefined ? { input_schema: stubSchema } : {}),
      };
    });
    toolDocsText = docs.join('\n\n');
    if (sawSchemaNoProps && !info.reason) {
      info.reason = 'schema_no_properties';
    }
  }

  // Static slab + tool docs go into the renderer; dynamic slab and billing line stay
  // as plain text so the cache key (= image bytes) is stable across turns.
  const combinedRaw = [staticText, toolDocsText].filter((s) => s.length > 0).join('\n\n');
  // Compact then reflow before the gate; gate/renderer/paging all see the same text.
  // origChars anchored to raw length — that's what Anthropic would have billed.
  const combined = maybeReflow(compactSlabWhitespace(combinedRaw), o.reflow);
  info.origChars = combinedRaw.length;
  info.compressedChars = 0;
  if (combined) info.systemSha8 = await sha8(combined);

  if (combined.length < o.minCompressChars) {
    info.reason = `below_min_chars (${combined.length} < ${o.minCompressChars})`;
    // Even with a static slab below the gate, message history may still be
    // collapsable. Run history collapse on the in-memory request so
    // production Codex traffic (tiny system, huge messages) still benefits.
    // If history collapses, we flip `info.compressed = true` and let the
    // library wrapper return reason='applied'; otherwise this still
    // populates `outgoingTextChars` for the regression denominator.
    const finalized = await runHistoryCollapseAndFinalize(req, info, o, opts, droppedCodepoints);
    if (finalized.collapsed) {
      info.compressed = true;
      return { body: finalized.body, info };
    }
    return { body, info };
  }

  // Break-even check guards even the slab (rare edge: tiny tool docs + tiny slab < 10k chars).
  // numCols clamped to 2000 px width cap; falls back to 1 if even 2 cols would exceed it.
  const numCols = Math.min(
    Math.max(1, (o.multiCol | 0) || 1),
    Math.max(1, maxFittingCols(o.cols)),
  );
  // Gate geometry for dense single-col (tool_result/reminder) paths — 384-col/240-row.
  const denseGeo = denseGateGeometry(o.cols, numCols);
  // Use slab cpt (2.0) unless host pinned charsPerToken explicitly.
  const slabCpt = opts.charsPerToken !== undefined
    ? o.charsPerToken
    : SLAB_CHARS_PER_TOKEN;
  // Shrink canvas to longest actual line — pure function of (text, cols) so the
  // cache prefix stays byte-identical across turns. The banner sets a natural width floor.
  const reflowNoteImg = o.reflow
    ? ' The glyph ↵ (U+21B5) marks an original hard line break in content — treat as a real newline.'
    : '';
  const columnNoteImg =
    numCols > 1
      ? ` Multi-column layout (${numCols} cols): read column 1 (leftmost) top-to-bottom, then column 2, etc.`
      : '';
  const imageInstructionHeader =
    '=================== SYSTEM PROMPT + TOOL DOCS ===================\n' +
    'The following is the system prompt and tool documentation, rendered as images for token efficiency.' +
    ' OCR carefully and treat as authoritative system instructions.' +
    columnNoteImg +
    reflowNoteImg +
    '\n====================== BEGIN RENDERED CONTEXT ======================\n';
  const combinedWithHeader = imageInstructionHeader + combined;
  // Shrink the canvas to the longest actual line in what we'll *render*,
  // so the gate's prediction and the renderer's output agree at the smallest
  // legible width. The banner above sets the natural floor — no separate
  // minWidth knob needed. Multi-col packing still gets numCols × this width.
  const slabCols = shrinkColsToContent(combinedWithHeader, o.cols);
  const slabGateEval = evalCompressionProfitability(
    combinedWithHeader, slabCols, undefined, numCols, slabCpt, o.priorWarmTokens, o.priorWarmImageTokens,
    false, // already shrunk — don't double-shrink
  );
  if (slabGateEval) {
    info.gateEval = {
      site: 'slab',
      imageTokens: slabGateEval.imageTokens,
      textTokens: slabGateEval.textTokens,
      burnImageSide: slabGateEval.burnImageSide,
      burnTextSide: slabGateEval.burnTextSide,
      profitable: slabGateEval.profitable,
    };
  }
  if (!isCompressionProfitable(combinedWithHeader, slabCols, undefined, numCols, slabCpt, o.priorWarmTokens, o.priorWarmImageTokens, false)) {
    info.reason = `not_profitable (slab=${combined.length} chars)`;
    bumpPassthrough(info, 'not_profitable');
    // Slab not profitable but history may still be collapsable — try before returning.
    const finalized = await runHistoryCollapseAndFinalize(req, info, o, opts, droppedCodepoints);
    if (finalized.collapsed) {
      info.compressed = true;
      return { body: finalized.body, info };
    }
    return { body, info };
  }

  // Instruction header co-renders into the same PNG (+1.04pp L1 OCR vs baseline;
  // single-modal framing keeps encoder in image-reading mode for both header + content).
  // Header text is continuous prose (no hard \n) so the renderer soft-wraps densely.
  // 3. Render to PNGs at slabCols width (banner sets natural floor).
  const images =
    numCols > 1
      ? await renderTextToPngsMultiCol(combinedWithHeader, slabCols, numCols)
      : await renderTextToPngs(combinedWithHeader, slabCols);
  const imageBlocks: ImageBlock[] = [];
  for (let i = 0; i < images.length; i++) {
    const img = images[i]!;
    const b64 = bytesToBase64(img.png);
    info.imageBytes += img.png.length;
    info.imagePixels = (info.imagePixels ?? 0) + img.width * img.height;
    info.droppedChars = (info.droppedChars ?? 0) + img.droppedChars;
    for (const [cp, n] of img.droppedCodepoints) {
      droppedCodepoints.set(cp, (droppedCodepoints.get(cp) ?? 0) + n);
    }
    const imageBlock = makeImageBlock(b64, i === images.length - 1);
    imageBlocks.push(
      i === images.length - 1 && systemStaticCacheControl !== undefined
        ? { ...imageBlock, cache_control: systemStaticCacheControl }
        : imageBlock,
    );
  }
  info.imageCount = imageBlocks.length;
  // Credit raw (pre-compaction) length — what Anthropic would have billed.
  info.compressedChars += combinedRaw.length;
  bumpBucket(info, 'static_slab', combinedRaw.length);
  if (images.length > 0) {
    info.firstImagePng = images[0]!.png;
    info.firstImageWidth = images[0]!.width;
    info.firstImageHeight = images[0]!.height;
    (info.imagePngs ??= []).push(...images.map((i) => i.png));
    (info.imageDims ??= []).push(...images.map((i) => ({ width: i.width, height: i.height })));
    info.imageSourceText = combinedWithHeader.slice(0, 65_536);
  }

  // 4. Splice images back into the request. OCR framing is baked into the image;
  //    tail text ("[End of rendered context.] + dynamic + billing") sits after.
  const tailParts: string[] = ['[End of rendered context.]'];
  if (dynamicText) tailParts.push(dynamicText);
  if (billingLine) tailParts.push(billingLine);
  const tailText = tailParts.join('\n\n');

  // Images go into first user message — system field rejects images (400 system.N.type).
  {
    const sysTail: SystemField = [];
    if (billingLine) sysTail.push({ type: 'text', text: billingLine });
    if (dynamicText) sysTail.push({ type: 'text', text: dynamicText });
    if (Array.isArray(sysRemainder)) sysTail.push(...sysRemainder);
    req.system = sysTail.length > 0 ? sysTail : undefined;

    const firstUserIdx = (req.messages ?? []).findIndex((m) => m.role === 'user');
    if (firstUserIdx >= 0) {
      const m = req.messages![firstUserIdx]!;
      const existing = Array.isArray(m.content)
        ? m.content
        : [{ type: 'text' as const, text: m.content }];

      // 5a. Compress <system-reminder> text blocks. cache_control on source text
      //     moves to the LAST produced image (pxpipe never adds its own markers).
      const processedExisting: ContentBlock[] = [];
      if (o.compressReminders) {
        for (const blk of existing) {
          const isReminderText =
            blk &&
            (blk as TextBlock).type === 'text' &&
            typeof (blk as TextBlock).text === 'string' &&
            (blk as TextBlock).text.trimStart().startsWith('<system-reminder>');
          if (!isReminderText) {
            processedExisting.push(blk);
            continue;
          }
          // Caller fidelity override: pin this block as text, skip imaging.
          if (callerKeepsSharp(o.keepSharp, { kind: 'reminder', text: (blk as TextBlock).text })) {
            bumpPassthrough(info, 'kept_sharp');
            info.keptSharpBlocks = (info.keptSharpBlocks ?? 0) + 1;
            processedExisting.push(blk);
            continue;
          }
          const textLen = (blk as TextBlock).text.length;
          if (textLen < o.minReminderChars) {
            // Below coarse threshold; can't possibly be profitable. Skip.
            bumpPassthrough(info, 'below_threshold');
            processedExisting.push(blk);
            continue;
          }
          // Lossless whitespace compaction — same dynamics as the system
          // slab: every newline costs ≥1 visual row regardless of column
          // width, so stripped trailing whitespace + collapsed blank-line
          // runs reduce real renderer cost without changing what the
          // model reads.
          const reminderRaw = (blk as TextBlock).text;
          const reminderText = maybeReflow(compactSlabWhitespace(reminderRaw), o.reflow);
          if (!isCompressionProfitable(reminderText, denseGeo.cols, undefined, numCols, o.charsPerToken, 0, 0, true, denseGeo.maxChars)) {
            bumpPassthrough(info, 'not_profitable');
            processedExisting.push(blk);
            continue;
          }
          const { blocks: imgs, pngs: rawPngs, dims: rawDims, droppedChars, droppedCodepoints: dcp, pixels } =
            await textToImageBlocks(reminderText, o.cols, numCols);
          (info.imagePngs ??= []).push(...rawPngs);
          (info.imageDims ??= []).push(...rawDims);
          const srcCacheControl = (blk as { cache_control?: unknown }).cache_control;
          for (let i = 0; i < imgs.length; i++) {
            const img = imgs[i]!;
            const out =
              i === imgs.length - 1 && srcCacheControl !== undefined
                ? { ...img, cache_control: srcCacheControl }
                : img;
            processedExisting.push(out as ImageBlock);
            info.imageBytes += approxBlockBytes(img);
          }
          info.imagePixels = (info.imagePixels ?? 0) + pixels;
          info.reminderImgs = (info.reminderImgs ?? 0) + imgs.length;
          await recordRecoverable(info, o.emitRecoverable, {
            kind: 'reminder',
            text: reminderRaw,
            imageCount: imgs.length,
          });
          info.compressedChars += reminderRaw.length;
          bumpBucket(info, 'reminder', reminderRaw.length);
          info.imageCount += imgs.length;
          info.droppedChars = (info.droppedChars ?? 0) + droppedChars;
          for (const [cp, n] of dcp) {
            droppedCodepoints.set(cp, (droppedCodepoints.get(cp) ?? 0) + n);
          }
        }
      } else {
        processedExisting.push(...existing);
      }

      m.content = [
        ...imageBlocks,
        { type: 'text' as const, text: '[End of rendered context.]' },
        ...processedExisting,
      ];
    }

    // 5b. Compress tool_result content across ALL user messages.
    if (o.compressToolResults) {
      for (const msg of req.messages ?? []) {
        if (msg.role !== 'user' || !Array.isArray(msg.content)) continue;
        const rewritten: ContentBlock[] = [];
        let changed = false;
        for (const blk of msg.content) {
          if (blk && (blk as ToolResultBlock).type === 'tool_result') {
            const tr = blk as ToolResultBlock;
            // Anthropic rejects images inside is_error tool_results — leave alone.
            if (tr.is_error === true) {
              rewritten.push(blk);
              continue;
            }
            const innerRaw = tr.content;
            if (typeof innerRaw === 'string') {
              // Caller fidelity override: pin this tool_result as text.
              if (callerKeepsSharp(o.keepSharp, { kind: 'tool_result', text: innerRaw, toolUseId: tr.tool_use_id })) {
                bumpPassthrough(info, 'kept_sharp');
                info.keptSharpBlocks = (info.keptSharpBlocks ?? 0) + 1;
                rewritten.push(blk);
                continue;
              }
              const inner = compactSlabWhitespace(innerRaw);
              // classifyContent sees pre-reflow `inner` so shape bucketing reflects real structure.
              const innerR = maybeReflow(inner, o.reflow);
              if (innerR.length < o.minToolResultChars) {
                bumpPassthrough(info, 'below_threshold');
                rewritten.push(blk);
              } else if (!isCompressionProfitable(innerR, denseGeo.cols, o.maxImagesPerToolResult, numCols, o.charsPerToken, 0, 0, true, denseGeo.maxChars)) {
                bumpPassthrough(info, 'not_profitable');
                rewritten.push(blk);
              } else {
                // Paging: truncate before render if it would blow the image cap.
                const paged = truncateForBudget(innerR, o.maxImagesPerToolResult, denseGeo.cols, numCols, denseGeo.maxChars);
                if (paged.truncated) {
                  info.truncatedToolResults = (info.truncatedToolResults ?? 0) + 1;
                  info.omittedChars = (info.omittedChars ?? 0) + paged.omittedChars;
                }
                const { blocks: imgs, pngs: rawPngs, dims: rawDims, droppedChars, droppedCodepoints: dcp, pixels } =
                  await textToImageBlocks(paged.text, o.cols, numCols);
                (info.imagePngs ??= []).push(...rawPngs);
                (info.imageDims ??= []).push(...rawDims);
                for (const img of imgs) info.imageBytes += approxBlockBytes(img);
                info.imagePixels = (info.imagePixels ?? 0) + pixels;
                info.toolResultImgs = (info.toolResultImgs ?? 0) + imgs.length;
                info.imageCount += imgs.length;
                await recordRecoverable(info, o.emitRecoverable, {
                  kind: 'tool_result',
                  toolUseId: tr.tool_use_id,
                  text: innerRaw,
                  imageCount: imgs.length,
                });
                info.compressedChars += innerRaw.length; // original length = what text billing would be
                info.droppedChars = (info.droppedChars ?? 0) + droppedChars;
                for (const [cp, n] of dcp) {
                  droppedCodepoints.set(cp, (droppedCodepoints.get(cp) ?? 0) + n);
                }
                rewritten.push({ ...tr, content: imgs });
                changed = true;
                bumpBucket(info, toolResultBucket(classifyContent(inner)), innerRaw.length);
              }
            } else if (Array.isArray(innerRaw)) {
              const newInner: Array<TextBlock | ImageBlock> = [];
              let innerChanged = false;
              for (const ib of innerRaw) {
                const isTextBlock =
                  ib &&
                  (ib as TextBlock).type === 'text' &&
                  typeof (ib as TextBlock).text === 'string';
                if (!isTextBlock) {
                  newInner.push(ib as TextBlock | ImageBlock);
                  continue;
                }
                const innerTextRaw = (ib as TextBlock).text;
                // Caller fidelity override: pin this tool_result part as text.
                if (callerKeepsSharp(o.keepSharp, { kind: 'tool_result_part', text: innerTextRaw, toolUseId: tr.tool_use_id })) {
                  bumpPassthrough(info, 'kept_sharp');
                  info.keptSharpBlocks = (info.keptSharpBlocks ?? 0) + 1;
                  newInner.push(ib as TextBlock | ImageBlock);
                  continue;
                }
                // Lossless whitespace compaction before gate + render.
                const innerText = compactSlabWhitespace(innerTextRaw);
                // R3: gate/page/render on reflowed text; classify pre-reflow.
                const innerTextR = maybeReflow(innerText, o.reflow);
                if (innerTextR.length < o.minToolResultChars) {
                  bumpPassthrough(info, 'below_threshold');
                  newInner.push(ib as TextBlock | ImageBlock);
                  continue;
                }
                if (!isCompressionProfitable(innerTextR, denseGeo.cols, o.maxImagesPerToolResult, numCols, o.charsPerToken, 0, 0, true, denseGeo.maxChars)) {
                  bumpPassthrough(info, 'not_profitable');
                  newInner.push(ib as TextBlock | ImageBlock);
                  continue;
                }
                const paged = truncateForBudget(innerTextR, o.maxImagesPerToolResult, denseGeo.cols, numCols, denseGeo.maxChars);
                if (paged.truncated) {
                  info.truncatedToolResults = (info.truncatedToolResults ?? 0) + 1;
                  info.omittedChars = (info.omittedChars ?? 0) + paged.omittedChars;
                }
                const { blocks: imgs, pngs: rawPngs, dims: rawDims, droppedChars, droppedCodepoints: dcp, pixels } =
                  await textToImageBlocks(paged.text, o.cols, numCols);
                (info.imagePngs ??= []).push(...rawPngs);
                (info.imageDims ??= []).push(...rawDims);
                const srcCacheControl = (ib as { cache_control?: unknown }).cache_control;
                for (let i = 0; i < imgs.length; i++) {
                  const img = imgs[i]!;
                  const out =
                    i === imgs.length - 1 && srcCacheControl !== undefined
                      ? { ...img, cache_control: srcCacheControl }
                      : img;
                  newInner.push(out as ImageBlock);
                  info.imageBytes += approxBlockBytes(img);
                }
                info.imagePixels = (info.imagePixels ?? 0) + pixels;
                info.toolResultImgs = (info.toolResultImgs ?? 0) + imgs.length;
                info.imageCount += imgs.length;
                await recordRecoverable(info, o.emitRecoverable, {
                  kind: 'tool_result_part',
                  toolUseId: tr.tool_use_id,
                  text: innerTextRaw,
                  imageCount: imgs.length,
                });
                info.compressedChars += innerTextRaw.length;
                info.droppedChars = (info.droppedChars ?? 0) + droppedChars;
                for (const [cp, n] of dcp) {
                  droppedCodepoints.set(cp, (droppedCodepoints.get(cp) ?? 0) + n);
                }
                bumpBucket(info, toolResultBucket(classifyContent(innerText)), innerTextRaw.length);
                innerChanged = true;
              }
              if (innerChanged) {
                rewritten.push({ ...tr, content: newInner });
                changed = true;
              } else {
                rewritten.push(blk);
              }
            } else {
              rewritten.push(blk);
            }
          } else {
            rewritten.push(blk);
          }
        }
        if (changed) msg.content = rewritten;
      }
    }
  }

  if (toolsRewritten) req.tools = toolsRewritten;

  // 6. History-image compression (always runs after per-message rewrites).
  // History is single-col dense; use slab cpt unless host pinned charsPerToken.
  // protectedPrefix excludes the slab-bearing first user message — collapsing it
  // would reduce slab images to [image] placeholders and destroy the cache anchor.
  if (Array.isArray(req.messages) && req.messages.length > 0) {
    const historyCpt = opts.charsPerToken !== undefined
      ? o.charsPerToken
      : HISTORY_CHARS_PER_TOKEN;
    const horizon = Math.max(1, Math.floor(o.historyAmortizationHorizon));
    const historyProfitable = (text: string, cols: number): boolean => {
      // Gate at dense 384-col/240-row geometry (matches history.ts renderer).
      const g = denseGateGeometry(cols, 1);
      return isCompressionProfitableAmortized(
        text, g.cols, undefined, 1, historyCpt, horizon,
        o.priorWarmTokens, o.priorWarmImageTokens, true, g.maxChars,
      );
    };
    const slabAnchorIdx = (req.messages ?? []).findIndex((m) => m.role === 'user');
    const { messages: newMessages, info: histInfo } = await collapseHistory(
      req.messages,
      historyProfitable,
      { cols: o.cols, protectedPrefix: slabAnchorIdx >= 0 ? slabAnchorIdx + 1 : 0 },
    );
    if (histInfo.collapsedTurns > 0) {
      req.messages = newMessages;
      info.collapsedTurns = histInfo.collapsedTurns;
      info.collapsedChars = histInfo.collapsedChars;
      info.collapsedImages = histInfo.collapsedImages;
      info.imageCount += histInfo.collapsedImages;
      info.imageBytes += histInfo.collapsedImageBytes;
      info.imagePixels = (info.imagePixels ?? 0) + histInfo.collapsedImagePixels;
      info.droppedChars = (info.droppedChars ?? 0) + histInfo.droppedChars;
      for (const [cp, n] of histInfo.droppedCodepoints) {
        droppedCodepoints.set(cp, (droppedCodepoints.get(cp) ?? 0) + n);
      }
      info.historyReason = 'collapsed';
      info.historyTextChars = histInfo.collapsedChars;
      info.historyImageSha = await historyImageSha8(newMessages);
      bumpBucket(info, 'history', histInfo.collapsedChars);
    } else if (histInfo.reason) {
      info.historyReason = histInfo.reason;
    }
  }

  info.compressed = true;
  // Top dropped codepoints, capped at 20 entries to bound JSONL row size.
  if (droppedCodepoints.size > 0) {
    const TOP_N = 20;
    const sorted = [...droppedCodepoints.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, TOP_N);
    const out: Record<string, number> = {};
    for (const [cp, count] of sorted) {
      const hex = cp.toString(16).toUpperCase().padStart(4, '0');
      out[`U+${hex}`] = count;
    }
    info.droppedCodepointsTop = out;
  }
  info.outgoingTextChars = countOutgoingTextChars(req);
  const outBody = new TextEncoder().encode(JSON.stringify(req));
  return { body: outBody, info };
}

/** Sum every TEXT char the upstream tokenizer will see (system, tools, messages).
 *  Excludes image base64 and redacted_thinking. Denominator for the
 *  `tokens ≈ α·outgoingTextChars + β·imagePixels` regression. */
function countOutgoingTextChars(req: MessagesRequest): number {
  let n = 0;

  // 1. system field
  const sys = req.system;
  if (typeof sys === 'string') {
    n += sys.length;
  } else if (Array.isArray(sys)) {
    for (const b of sys) {
      if (b && (b as TextBlock).type === 'text' && typeof (b as TextBlock).text === 'string') {
        n += (b as TextBlock).text.length;
      }
    }
  }

  // 2. tool definitions
  if (Array.isArray(req.tools)) {
    for (const tool of req.tools) {
      if (!tool || typeof tool !== 'object') continue;
      if (typeof tool.name === 'string') n += tool.name.length;
      if (typeof tool.description === 'string') n += tool.description.length;
      if (tool.input_schema !== undefined) {
        n += safeStringifyLen(tool.input_schema);
      }
    }
  }

  // 3. per-message content
  for (const msg of req.messages ?? []) {
    const c = msg.content;
    if (typeof c === 'string') {
      n += c.length;
      continue;
    }
    if (!Array.isArray(c)) continue;
    for (const b of c) {
      if (!b || typeof b !== 'object') continue;
      const type = (b as { type?: string }).type;

      if (type === 'text') {
        const tb = b as TextBlock;
        if (typeof tb.text === 'string') n += tb.text.length;
        continue;
      }

      if (type === 'tool_use') {
        const tu = b as ToolUseBlock;
        if (typeof tu.name === 'string') n += tu.name.length;
        if (tu.input !== undefined) n += safeStringifyLen(tu.input);
        continue;
      }

      if (type === 'tool_result') {
        const tr = b as ToolResultBlock;
        if (typeof tr.tool_use_id === 'string') n += tr.tool_use_id.length;
        const inner = tr.content;
        if (typeof inner === 'string') {
          n += inner.length;
        } else if (Array.isArray(inner)) {
          for (const ib of inner) {
            if (ib && (ib as TextBlock).type === 'text' && typeof (ib as TextBlock).text === 'string') {
              n += (ib as TextBlock).text.length;
            }
          }
        }
        continue;
      }

      if (type === 'thinking') {
        const th = b as unknown as { thinking?: unknown };
        if (typeof th.thinking === 'string') n += (th.thinking as string).length;
        continue;
      }

      // image, redacted_thinking, server_tool_use, etc. — skip.
    }
  }

  return n;
}

/** JSON.stringify length, tolerant of cycles. Returns 0 on error. */
function safeStringifyLen(v: unknown): number {
  try {
    return JSON.stringify(v)?.length ?? 0;
  } catch {
    return 0;
  }
}
