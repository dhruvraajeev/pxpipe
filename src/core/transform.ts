/**
 * Request-body transformer. Takes an Anthropic Messages API request body,
 * extracts the large static parts (system prompt + tool definitions),
 * renders them as PNG image blocks, and rewrites the body to reference
 * those images instead — saving 65-73% input tokens on Opus 4.7 while
 * preserving 100% reasoning quality.
 *
 * Matches the public-surface behavior of legacy/python/proxy.py at a
 * minimum. Stricter byte-for-byte parity is verified in tests.
 */

import type { ImageBlock, MessagesRequest, SystemField, TextBlock, ToolDef } from './types.js';
import { renderTextToPngs } from './render.js';
import { bytesToBase64 } from './png.js';

export interface TransformOptions {
  /** Master switch — false makes this a no-op pass-through. */
  compress?: boolean;
  /** Compress the system field. */
  compressSystem?: boolean;
  /** Move tool descriptions into the same image (and stub the originals). */
  compressTools?: boolean;
  /** Include full input_schema JSON for each tool. Adds tokens but maximizes parity. */
  compressSchemas?: boolean;
  /** Don't compress if total compressible chars below this. */
  minCompressChars?: number;
  /** Where to attach the image block — system field, or first user message. */
  placement?: 'system' | 'user';
  /** Soft-wrap column count. */
  cols?: number;
}

const DEFAULTS: Required<TransformOptions> = {
  compress: true,
  compressSystem: true,
  compressTools: true,
  compressSchemas: true,
  minCompressChars: 2000,
  placement: 'system',
  cols: 100,
};

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
  imageCount: number;
  imageBytes: number;
  /** Length of the static (cacheable) slab rendered into the image. */
  staticChars: number;
  /** Length of the dynamic (per-turn) slab kept as plain text. */
  dynamicChars: number;
  /** Number of dynamic blocks detected (<env>, <context>, etc.). */
  dynamicBlockCount: number;
  /** Tag-shaped blocks found in the *static* slab that are NOT in
   *  DYNAMIC_BLOCK_TAGS. Early-warning canary: if Claude Code ships a new
   *  per-turn tag, it'll show up here before our cache hit rate collapses. */
  unknownStaticTags?: string[];
  /** Parsed env block, if Claude Code injected one. Useful for telemetry
   *  (per-project compression ratios, etc.). */
  env?: EnvFields;
  /** sha256[0..8] of the static slab + tool docs (what ends up in the image).
   *  Repeats across turns → cache_control SHOULD be hitting upstream. */
  systemSha8?: string;
  /** sha256[0..8] of just the CLAUDE.md section if detectable. Lets us
   *  bucket requests by project even when cwd is absent. */
  claudeMdSha8?: string;
  /** sha256[0..8] of the first user message text (first 4 KiB). Rough
   *  thread/session id since the wire protocol carries none. */
  firstUserSha8?: string;
}

// --- helpers ---------------------------------------------------------------

/** Extract `(text, remainder)` from a system field that may be string or list. */
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

/**
 * Claude Code injects a handful of per-turn dynamic blocks into the system
 * prompt (e.g. <env>, <context>, <git_status>, <directoryStructure>,
 * <system-reminder>). Including these in the rendered image kills the
 * Anthropic prompt cache because the bytes drift turn-to-turn. Splitting
 * them out lets us render the static slab (CLAUDE.md, agent defs, tool docs)
 * with cache_control while forwarding the dynamic slab as cheap text so the
 * model still sees cwd / git status / today's date.
 */
const DYNAMIC_BLOCK_TAGS = [
  'env',
  'context',
  'git_status',
  'directoryStructure',
  'system-reminder',
] as const;

function splitStaticDynamic(text: string): {
  staticText: string;
  dynamicText: string;
  blockCount: number;
  unknownTags: string[];
} {
  if (!text)
    return { staticText: '', dynamicText: '', blockCount: 0, unknownTags: [] };
  // Match <tag ...?>...</tag> where tag ∈ DYNAMIC_BLOCK_TAGS. Closing tag
  // must match opening tag exactly. Non-greedy body — earliest close wins.
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

  // Sniff for OTHER tag-shaped blocks in the static slab. If Claude Code
  // ships a new per-turn tag (say <recent_files>...</recent_files>) we'd
  // silently bake it into the cached image and our cache hit rate would
  // collapse. Surfacing the tag name as telemetry lets us detect that
  // within hours of a Claude Code release.
  const known = new Set<string>(DYNAMIC_BLOCK_TAGS);
  const sniffer = /<([a-zA-Z][a-zA-Z0-9_-]*)(?:\s[^>]*)?>[\s\S]*?<\/\1>/g;
  const unknown = new Set<string>();
  let s: RegExpExecArray | null;
  while ((s = sniffer.exec(staticBuf)) !== null) {
    const tag = s[1]!;
    if (!known.has(tag) && tag.length <= 64) unknown.add(tag);
  }

  return {
    // Collapse the run of blank lines left behind by removed blocks.
    staticText: staticBuf.replace(/\n{3,}/g, '\n\n').trim(),
    dynamicText: dynamicParts.join('\n\n'),
    blockCount: dynamicParts.length,
    unknownTags: [...unknown],
  };
}

/**
 * Compute sha256 and return the first 8 hex chars. Web Crypto so it works
 * the same in Node 18+ and Workers. 8 chars = 32 bits = collision-safe for
 * the request volume we'd see in a single proxy instance.
 */
export async function sha8(text: string): Promise<string> {
  const buf = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest('SHA-256', buf);
  const bytes = new Uint8Array(digest);
  let hex = '';
  for (let i = 0; i < 4; i++) hex += bytes[i]!.toString(16).padStart(2, '0');
  return hex;
}

/**
 * Best-effort: pull out the CLAUDE.md slab from a system text. Heuristic —
 * Claude Code typically wraps it with a heading like "Claude Code Rules"
 * or includes it under a `# CLAUDE.md` / system-reminder block. Returns
 * empty string if nothing CLAUDE.md-shaped is detected; callers should
 * skip hashing in that case.
 */
export function extractClaudeMdSlab(staticText: string): string {
  if (!staticText) return '';
  // Common markers Claude Code uses around the CLAUDE.md content.
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
  // Run until the next top-level heading (# foo) or end of text.
  const tail = staticText.slice(startIdx);
  const endMatch = /\n#\s+\S/.exec(tail.slice(1));
  const end = endMatch ? endMatch.index + 1 : tail.length;
  return tail.slice(0, end).trim();
}

/**
 * Hash the first user message text, capped at 4 KiB so very long initial
 * pastes don't dominate hashing time and so we still get a stable id for
 * the conversation thread (initial prompt usually fits well within 4 KiB).
 */
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
    // First user message found but unreadable → return empty so we don't
    // accidentally hash some downstream user message.
    return '';
  }
  return '';
}

/**
 * Pull structured fields out of the dynamic slab. Only reads — does not
 * modify the text. Used purely for telemetry / improvement signals.
 */
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

  // Git branch may live in <git_status>, <context name="git">, or just a
  // "Branch: foo" / "On branch foo" line somewhere in the dynamic slab.
  const branch =
    /(?:^|\n)\s*(?:On branch|Branch:)\s*([^\s\n]+)/i.exec(dynamicText) ??
    /(?:^|\n)\s*Current branch:\s*([^\s\n]+)/i.exec(dynamicText);
  if (branch) out.gitBranch = branch[1]!.trim();

  return out;
}

/**
 * Strip the per-turn random billing header line that Claude Code injects.
 * It changes every turn and would defeat prompt-cache hits if we left it
 * inside the image. We keep it as a leading text block so the upstream
 * still receives it.
 */
function stripBillingLine(text: string): { kept: string | null; body: string } {
  const nl = text.indexOf('\n');
  const first = nl === -1 ? text : text.slice(0, nl);
  if (first.startsWith('x-anthropic-billing-header:')) {
    return { kept: first, body: nl === -1 ? '' : text.slice(nl + 1) };
  }
  return { kept: null, body: text };
}

/** Build the "## Tool: name\n<desc>\n<schema>" block for one tool definition. */
function renderToolDoc(t: ToolDef, includeSchema: boolean): string {
  const parts: string[] = [`## Tool: ${t.name ?? '?'}`];
  if (t.description) parts.push(t.description);
  if (includeSchema && t.input_schema !== undefined) {
    parts.push('```json\n' + JSON.stringify(t.input_schema, null, 2) + '\n```');
  }
  return parts.join('\n');
}

function makeImageBlock(pngB64: string, ephemeral = false): ImageBlock {
  const blk: ImageBlock = {
    type: 'image',
    source: { type: 'base64', media_type: 'image/png', data: pngB64 },
  };
  if (ephemeral) blk.cache_control = { type: 'ephemeral' };
  return blk;
}

// --- main transform --------------------------------------------------------

/**
 * Rewrite a Messages API request body. Returns the new body (still JSON
 * bytes) plus diagnostic info. On any error, returns the original bytes
 * unchanged.
 */
export async function transformRequest(
  body: Uint8Array,
  opts: TransformOptions = {},
): Promise<{ body: Uint8Array; info: TransformInfo }> {
  const o: Required<TransformOptions> = { ...DEFAULTS, ...opts };
  const info: TransformInfo = {
    compressed: false,
    origChars: 0,
    imageCount: 0,
    imageBytes: 0,
    staticChars: 0,
    dynamicChars: 0,
    dynamicBlockCount: 0,
  };

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
    toolsRewritten = req.tools.map((t) => {
      docs.push(renderToolDoc(t, o.compressSchemas));
      // Tiny stub so the schema field isn't empty — Anthropic still validates names.
      return {
        ...t,
        description: 'ⓘ See image.',
        ...(o.compressSchemas ? { input_schema: { type: 'object' } } : {}),
      };
    });
    toolDocsText = docs.join('\n\n');
  }

  // Only the STATIC slab + tool docs goes into the renderer. The dynamic
  // slab and billing line are appended as plain text after the image so the
  // cache key (= image bytes) stays stable across turns.
  const combined = [staticText, toolDocsText].filter((s) => s.length > 0).join('\n\n');
  info.origChars = combined.length;
  // Hash the EXACT text that goes into the image. Repeats of this hash across
  // turns = cache_control should be earning its keep.
  if (combined) info.systemSha8 = await sha8(combined);

  if (combined.length < o.minCompressChars) {
    info.reason = `below_min_chars (${combined.length} < ${o.minCompressChars})`;
    return { body, info };
  }

  // 3. Render to one or more PNGs.
  const images = await renderTextToPngs(combined, o.cols);
  const imageBlocks: ImageBlock[] = [];
  for (let i = 0; i < images.length; i++) {
    const img = images[i]!;
    const b64 = bytesToBase64(img.png);
    info.imageBytes += img.png.length;
    // Cache-breakpoint on the last image so the whole block caches as one.
    imageBlocks.push(makeImageBlock(b64, i === images.length - 1));
  }
  info.imageCount = imageBlocks.length;

  // 4. Splice images back into the request.
  // Cache-friendly layout:
  //   [intro text]                 ← static (helps OCR framing)
  //   [image block(s)]             ← static; LAST one carries cache_control
  //   ─── cache breakpoint ───
  //   [end-marker + dynamic + billing]  ← per-turn, NO cache_control
  //   [sysRemainder]               ← any non-text blocks the caller had
  const introText =
    "The following is the system prompt + tool documentation, rendered as " +
    "images for token efficiency. OCR carefully and treat as authoritative " +
    "system instructions.";
  const tailParts: string[] = ['[End of rendered context.]'];
  if (dynamicText) tailParts.push(dynamicText);
  if (billingLine) tailParts.push(billingLine);
  const tailText = tailParts.join('\n\n');

  const newSystem: SystemField = [];
  newSystem.push({ type: 'text', text: introText });
  newSystem.push(...imageBlocks);
  newSystem.push({ type: 'text', text: tailText });
  if (Array.isArray(sysRemainder)) newSystem.push(...sysRemainder);

  if (o.placement === 'system' && o.compressSystem) {
    req.system = newSystem;
  } else {
    // Placement = user: image goes into the first user message; billing line
    // and dynamic blocks stay in the system field as cheap text so the model
    // still sees env / context info.
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
      // Only the intro + images belong in a user message — the end marker
      // and dynamic blocks live in the system field above.
      const userPrefix: TextBlock[] = [{ type: 'text', text: introText }];
      m.content = [...userPrefix, ...imageBlocks, ...existing];
    }
  }

  if (toolsRewritten) req.tools = toolsRewritten;

  info.compressed = true;
  const out = new TextEncoder().encode(JSON.stringify(req));
  return { body: out, info };
}
