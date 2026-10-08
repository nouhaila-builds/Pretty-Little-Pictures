export function formatAp(value: number | null) {
  if (value === null || Number.isNaN(value)) return "—";
  if (value === 0) return "0";
  if (value < 0.001) return value.toExponential(1);
  return value.toFixed(3);
}
