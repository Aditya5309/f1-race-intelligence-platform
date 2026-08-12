import { Skeleton } from "./Skeleton";

/** Matches TrendLineChart's fixed h-52 container exactly. */
export function SkeletonTrendChart({
  label = "Loading trend chart…",
}: {
  label?: string;
}) {
  return (
    <div role="status" aria-live="polite" className="h-52">
      <span className="sr-only">{label}</span>
      <Skeleton className="h-full w-full" />
    </div>
  );
}
