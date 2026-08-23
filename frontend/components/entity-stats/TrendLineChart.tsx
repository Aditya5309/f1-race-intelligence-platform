"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useChartReveal } from "@/components/motion";

/**
 * Shared trend-line chart — Driver Explorer/Team's qualifying/finishing/
 * points/win-probability trends. `invertY` renders position charts with
 * P1 at the top (lower position number = better), matching
 * app/views/charts.py::trend_line's own invert_y option.
 *
 * Viewport-triggered (frontend-animation-visibility-audit.md P1 fix):
 * Driver Explorer's Trends section confirmed below the fold on initial
 * load, so this chart used to finish its Recharts draw-in animation before
 * the user scrolled to it. `ref` is attached to the height-reserving
 * wrapper (same convention SkeletonTrendChart already uses to match this
 * chart's footprint) so nothing shifts when the real chart mounts; the
 * `ResponsiveContainer` itself is only rendered once `shouldRender` is
 * true, which — combined with Recharts having no server-rendered fallback
 * — is what actually defers the draw-in animation, not just its
 * `isAnimationActive` flag (that flag alone doesn't stop Recharts from
 * mounting fully-drawn immediately). Team's usage lands above the fold in
 * every case checked in the audit, so this is a no-op there — `useInView`
 * fires immediately for an already-visible element.
 */
export function TrendLineChart({
  data,
  xKey,
  yKey,
  color,
  invertY = false,
  emptyMessage,
  title,
}: {
  data: Record<string, unknown>[];
  xKey: string;
  yKey: string;
  color: string;
  invertY?: boolean;
  emptyMessage: string;
  title?: string;
}) {
  const { isAnimationActive, ref, shouldRender } = useChartReveal({ viewport: true });
  if (!data.length) {
    return <p className="text-sm text-[var(--color-muted-text)]">{emptyMessage}</p>;
  }
  return (
    <div>
      {title && (
        <p className="label-caps mb-1 text-[10px] text-[var(--color-muted-text)]">{title}</p>
      )}
      <div ref={ref} className="h-52">
        {shouldRender && (
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
                isAnimationActive={isAnimationActive}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
