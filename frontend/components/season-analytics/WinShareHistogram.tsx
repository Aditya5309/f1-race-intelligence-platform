"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useChartReveal } from "@/components/motion";
import type { HistogramBin } from "@/lib/season-aggregation";

/**
 * Viewport-triggered (frontend-animation-visibility-audit.md P1 fix): the
 * 7th section on Season Analytics, confirmed below the fold and already
 * fully settled (off-screen) by the time this page's own audit check ran.
 * `AccuracyChart` (4th section, same page) was confirmed already above the
 * fold and is deliberately left untouched — see that file.
 */
export function WinShareHistogram({ bins }: { bins: HistogramBin[] }) {
  const { isAnimationActive, ref, shouldRender } = useChartReveal({ viewport: true });
  return (
    <div ref={ref} className="h-52">
      {shouldRender && (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bins}>
            <CartesianGrid stroke="var(--color-hairline)" vertical={false} />
            <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
            <Tooltip />
            <Bar
              dataKey="count"
              fill="var(--color-accent-blue)"
              radius={[3, 3, 0, 0]}
              isAnimationActive={isAnimationActive}
            />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
