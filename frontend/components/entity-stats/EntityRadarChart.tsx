"use client";

import { PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer } from "recharts";
import { useChartReveal } from "@/components/motion";

export interface RadarSeries {
  label: string;
  color: string;
  scores: Record<string, number>;
}

/**
 * Season-profile radar — one or two overlaid series (Driver Explorer:
 * one driver; Compare Drivers: two). Scores are 0-100, normalized within
 * one season's field only (see app/entity_stats.py::radar_scores) — never
 * comparable across seasons, callers must say so in the UI themselves.
 *
 * Viewport-triggered (frontend-animation-visibility-audit.md P1/P2 fix):
 * on Driver Explorer this renders below the already-confirmed-below-fold
 * Trends section, so it's below the fold too; on Compare Drivers the audit
 * found it straddling the fold (partial evidence). See TrendLineChart's
 * doc comment for why gating `shouldRender` (not just `isAnimationActive`)
 * is what actually defers the draw-in.
 */
export function EntityRadarChart({ series }: { series: RadarSeries[] }) {
  const { isAnimationActive, ref, shouldRender } = useChartReveal({ viewport: true });
  if (!series.length) return null;
  const axes = Object.keys(series[0].scores);
  const data = axes.map((axis) => {
    const row: Record<string, string | number> = { axis };
    for (const s of series) row[s.label] = s.scores[axis];
    return row;
  });
  return (
    <div ref={ref} className="h-72">
      {shouldRender && (
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data}>
            <PolarGrid stroke="var(--color-hairline)" />
            <PolarAngleAxis dataKey="axis" tick={{ fontSize: 11 }} />
            {series.map((s) => (
              <Radar
                key={s.label}
                name={s.label}
                dataKey={s.label}
                stroke={s.color}
                fill={s.color}
                fillOpacity={0.25}
                isAnimationActive={isAnimationActive}
              />
            ))}
          </RadarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
