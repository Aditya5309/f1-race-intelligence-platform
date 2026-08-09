import { COLUMN_CLASSES } from "@/components/StatRow";
import { Skeleton } from "./Skeleton";

/**
 * Mirrors StatRow's grid exactly — reuses the same COLUMN_CLASSES map so
 * the two can never drift out of sync, and `columns` is required (must
 * match the real StatRow's `columns` prop at each call site, or the
 * loading and loaded states would reflow to a different column count).
 */
export function SkeletonStatRow({
  columns,
  label = "Loading stats…",
}: {
  columns: 3 | 4 | 5 | 6;
  label?: string;
}) {
  return (
    <div role="status" aria-live="polite" className={`grid gap-4 ${COLUMN_CLASSES[columns]}`}>
      <span className="sr-only">{label}</span>
      {Array.from({ length: columns }).map((_, i) => (
        <div key={i} aria-hidden="true" className="rounded-xl bg-white/40 p-4">
          <Skeleton className="h-7 w-3/4" />
          <Skeleton className="mt-2 h-2.5 w-1/2" />
        </div>
      ))}
    </div>
  );
}
