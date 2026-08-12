import { Skeleton } from "./Skeleton";

/**
 * Generic row-based table placeholder. `hasHeader=true` (default) adds one
 * extra header-shaped row, matching ModelComparisonTable and Driver
 * Explorer's race log — both real `<table>`s with a `<thead>`. Race
 * Center's FullFieldTable is an `<ol>` of DriverRow items with no header
 * row at all, so its call site should pass `hasHeader={false}` — a fixed
 * `rows + 1` would overshoot FullFieldTable's real height by one row.
 */
export function SkeletonTable({
  rows = 5,
  columns = 3,
  hasHeader = true,
  label = "Loading table…",
}: {
  rows?: number;
  columns?: number;
  hasHeader?: boolean;
  label?: string;
}) {
  return (
    <div role="status" aria-live="polite" className="rounded-xl bg-white/40">
      <span className="sr-only">{label}</span>
      <div className="divide-y divide-[var(--color-hairline)]" aria-hidden="true">
        {Array.from({ length: hasHeader ? rows + 1 : rows }).map((_, r) => (
          <div key={r} className="flex items-center gap-4 px-3 py-2.5">
            {Array.from({ length: columns }).map((_, c) => (
              <Skeleton key={c} className="h-3 flex-1" />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
