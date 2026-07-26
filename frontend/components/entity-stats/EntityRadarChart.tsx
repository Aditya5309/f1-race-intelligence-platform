"use client";

import { PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer } from "recharts";

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
 */
export function EntityRadarChart({ series }: { series: RadarSeries[] }) {
  if (!series.length) return null;
  const axes = Object.keys(series[0].scores);
  const data = axes.map((axis) => {
    const row: Record<string, string | number> = { axis };
    for (const s of series) row[s.label] = s.scores[axis];
    return row;
  });
  return (
    <div className="h-72">
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
            />
          ))}
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
