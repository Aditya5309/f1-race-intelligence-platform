"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useChartReveal } from "@/components/motion";

/**
 * Shared trend-line chart — Driver Explorer/Team's qualifying/finishing/
 * points/win-probability trends. `invertY` renders position charts with
 * P1 at the top (lower position number = better), matching
 * app/views/charts.py::trend_line's own invert_y option.
 */
export function TrendLineChart({
  data,
  xKey,
  yKey,
  color,
  invertY = false,
  emptyMessage,
}: {
  data: Record<string, unknown>[];
  xKey: string;
  yKey: string;
  color: string;
  invertY?: boolean;
  emptyMessage: string;
}) {
  const chartReveal = useChartReveal();
  if (!data.length) {
    return <p className="text-sm text-[var(--color-muted-text)]">{emptyMessage}</p>;
  }
  return (
    <div className="h-52">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid stroke="var(--color-hairline)" vertical={false} />
          <XAxis dataKey={xKey} tick={{ fontSize: 10 }} />
          <YAxis reversed={invertY} tick={{ fontSize: 11 }} />
          <Tooltip />
          <Line
            type="monotone"
            dataKey={yKey}
            stroke={color}
            dot={{ r: 3 }}
            strokeWidth={2}
            {...chartReveal}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
