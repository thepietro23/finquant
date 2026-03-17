/** Indian number formatting + financial display utilities */

/** Format number in Indian Lakh/Crore system with INR symbol */
export function formatINR(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e7) return `${(value / 1e7).toFixed(2)} Cr`;
  if (abs >= 1e5) return `${(value / 1e5).toFixed(2)} L`;
  return new Intl.NumberFormat('en-IN', {
    style: 'currency', currency: 'INR', maximumFractionDigits: 0,
  }).format(value);
}

/** Format percentage with + sign for positive */
export function formatPct(value: number, decimals = 2): string {
  const sign = value > 0 ? '+' : '';
  return `${sign}${(value * 100).toFixed(decimals)}%`;
}

/** Format ratio (Sharpe, Sortino, etc.) */
export function formatRatio(value: number, decimals = 2): string {
  return value.toFixed(decimals);
}

/** Color class based on positive/negative value */
export function valueColor(value: number): string {
  if (value > 0) return 'text-profit';
  if (value < 0) return 'text-loss';
  return 'text-text-secondary';
}

/** Background class for value badges */
export function valueBg(value: number): string {
  if (value > 0) return 'bg-profit-light text-profit';
  if (value < 0) return 'bg-loss-light text-loss';
  return 'bg-bg-card text-text-secondary';
}
