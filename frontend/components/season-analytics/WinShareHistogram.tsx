"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useChartReveal } from "@/components/motion";
import type { HistogramBin } from "@/lib/season-aggregation";

export function WinShareHistogram({ bins }: { bins: HistogramBin[] }) {
  const chartReveal = useChartReveal();
  return (
    <div className="h-52">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={bins}>
          <CartesianGrid stroke="var(--color-hairline)" vertical={false} />
          <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="count" fill="var(--color-accent-blue)" radius={[3, 3, 0, 0]} {...chartReveal} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
