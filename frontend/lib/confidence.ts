/**
 * Ported verbatim from app/views/components.py's _CONFIDENCE_TIERS /
 * confidence_tier() — same thresholds/wording/star counts, so the two
 * frontends never drift on what "Strong favorite" vs "Wide open" means.
 * Keep in sync if that file changes.
 */
const CONFIDENCE_TIERS: readonly [threshold: number, label: string, stars: number][] = [
  [0.5, "Strong favorite", 5],
  [0.3, "Clear favorite", 4],
  [0.2, "Slight edge", 3],
  [0.0, "Wide open", 2],
];

export interface ConfidenceTier {
  label: string;
  stars: number;
}

export function confidenceTier(probability: number): ConfidenceTier {
  for (const [threshold, label, stars] of CONFIDENCE_TIERS) {
    if (probability >= threshold) return { label, stars };
  }
  const [, label, stars] = CONFIDENCE_TIERS[CONFIDENCE_TIERS.length - 1];
  return { label, stars };
}
