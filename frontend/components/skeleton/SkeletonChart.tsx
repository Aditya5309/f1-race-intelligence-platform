import { Skeleton } from "./Skeleton";

const BAR_HEIGHTS = ["h-1/3", "h-2/3", "h-1/2", "h-full", "h-3/4", "h-2/5"];

/**
 * Generic chart placeholder for non-trend charts (AccuracyChart's bar/line
 * pair, WinShareHistogram, EntityRadarChart). `variant` is purely a
 * cosmetic shimmer-shape choice — none of these real charts have a fixed
 * internal structure worth mirroring exactly, unlike TrendLineChart.
 */
export function SkeletonChart({
  height = "h-52",
  variant = "bars",
  label = "Loading chart…",
}: {
  height?: string;
  variant?: "bars" | "line" | "radar";
  label?: string;
}) {
  return (
    <div role="status" aria-live="polite" className={height}>
      <span className="sr-only">{label}</span>
      {variant === "radar" ? (
        <div className="flex h-full items-center justify-center">
          <Skeleton circle className="h-4/5 w-4/5 max-h-full max-w-full" />
        </div>
      ) : variant === "bars" ? (
        <div className="flex h-full items-end gap-3">
          {BAR_HEIGHTS.map((h, i) => (
            <Skeleton key={i} className={`w-full ${h}`} />
          ))}
        </div>
      ) : (
        <Skeleton className="h-full w-full" />
      )}
    </div>
  );
}
