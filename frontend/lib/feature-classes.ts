/**
 * Ported verbatim from app/views/insights.py's _STABLE_FEATURES /
 * _ERA_SENSITIVE_FEATURES / _EXPERIMENTAL_FEATURES — display copy only,
 * mirrored here (not fetched) for the same reason the Python source
 * mirrors src/features/metadata.py: the dashboard process cannot import
 * src/. Keep in sync on reclassification.
 */

import { color } from "./design-tokens";

export const STABLE_FEATURES = [
  "qualifying_position",
  "qualifying_gap_to_pole_pct",
  "reached_q2",
  "reached_q3",
  "pit_lane_start",
  "grid_adjusted",
  "grid_position_norm",
  "driver_experience_races",
  "driver_avg_finish_last_5",
  "driver_dnf_rate_last_5",
  "driver_standing_position_prev",
  "constructor_standing_position_prev",
];

export const ERA_SENSITIVE_FEATURES = [
  "driver_wins_last_3",
  "driver_wins_last_5",
  "driver_wins_last_10",
  "driver_podiums_last_5",
  "driver_points_last_5",
  "constructor_wins_last_3",
  "constructor_wins_last_5",
  "constructor_podiums_last_5",
  "constructor_dnf_rate_last_5",
  "driver_standing_points_prev",
  "driver_standing_wins_prev",
  "constructor_standing_points_prev",
];

export const EXPERIMENTAL_FEATURES = [
  "q1_sec",
  "q2_sec",
  "q3_sec",
  "driver_circuit_starts",
  "driver_circuit_wins",
  "driver_circuit_avg_finish",
  "constructor_circuit_wins",
];

/** Same three semantic colors as app/views/insights.py's st.badge() calls.
 * "stable" reuses design-tokens.ts's color.liveGreen (same value, single
 * source of truth); era_sensitive/experimental have no existing token
 * (same class of domain-specific addition as constructor-colors.ts). */
const FEATURE_CLASS_COLORS: Record<string, string> = {
  stable: color.liveGreen,
  era_sensitive: "#E08A3B",
  experimental: "#8B5CF6",
};
const DEFAULT_FEATURE_CLASS_COLOR = "#8A8272";

export function featureClassColor(featureClass: string): string {
  return FEATURE_CLASS_COLORS[featureClass] ?? DEFAULT_FEATURE_CLASS_COLOR;
}
