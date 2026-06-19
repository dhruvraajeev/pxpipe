/**
 * Cache-aware baseline math for the unproxied counterfactual.
 * Workers-safe: no node:, no Buffer, no process.*. Pure number math.
 * See docs/CACHING_AND_SAVINGS.md for the full derivation and audit history.
 */

/** Documented Anthropic price ratios: cc_5m = 1.25×, cr = 0.1× base input. One-line change if rates change. */
export const CACHE_CREATE_RATE = 1.25;
export const CACHE_READ_RATE = 0.1;

/**
 * Weighted input cost for the unproxied TEXT counterfactual (see docs/CACHING_AND_SAVINGS.md).
 * baseline_eff = cacheable×CACHE_READ_RATE + coldTail×1.0
 * Saving = baseline_eff − actual_eff; can be negative (honestly reported, not floored).
 *
 * @param baselineCacheable  tokens up to the last cache_control marker. ≤0 ⇒ credit nothing.
 */
export function computeBaselineInputEff(
  baseline: number,
  baselineCacheable: number,
  inputTokens: number,
  cc: number,
  cr: number,
): number {
  if (baseline <= 0) return 0;
  // Probe miss: can't split prefix from tail, so credit nothing (same as actual).
  if (baselineCacheable <= 0) return computeActualInputEff(inputTokens, cc, cr);
  const cacheable = Math.min(baselineCacheable, baseline);
  const coldTail = baseline - cacheable;
  return cacheable * CACHE_READ_RATE + coldTail * 1.0;
}

/** Weighted input cost pxpipe actually paid this turn. */
export function computeActualInputEff(
  inputTokens: number,
  cc: number,
  cr: number,
): number {
  return inputTokens + cc * CACHE_CREATE_RATE + cr * CACHE_READ_RATE;
}
