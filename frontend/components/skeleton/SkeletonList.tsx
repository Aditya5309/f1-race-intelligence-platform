import { Skeleton } from "./Skeleton";

/**
 * `variant="list"` (default) matches PredictionList/FullFieldTable's
 * DriverRow rows and RankedBar — a dot + label line + trailing value per
 * item, divided vertically. `variant="grid"` matches PodiumRow's actual
 * shape — a horizontal 3-column grid of bordered cards, not a vertical
 * list at all — these are structurally different real layouts, not one
 * shared list shape (same reasoning as SkeletonHero's feature/bare split).
 * Callers using "grid" for the podium slot should pass `items={3}` to
 * match PodiumRow's fixed top-3 real behavior.
 */
export function SkeletonList({
  items = 5,
  variant = "list",
  label = "Loading list…",
}: {
  items?: number;
  variant?: "list" | "grid";
  label?: string;
}) {
  if (variant === "grid") {
    return (
      <div role="status" aria-live="polite" className="grid gap-3 sm:grid-cols-3">
        <span className="sr-only">{label}</span>
        {Array.from({ length: items }).map((_, i) => (
          <div key={i} aria-hidden="true" className="rounded-xl bg-white/40 p-4">
            <Skeleton className="h-5 w-2/3" />
            <Skeleton className="mt-2 h-3 w-1/2" />
            <Skeleton className="mt-2 h-4 w-1/4" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div
      role="status"
      aria-live="polite"
      className="divide-y divide-[var(--color-hairline)]"
    >
      <span className="sr-only">{label}</span>
      {Array.from({ length: items }).map((_, i) => (
        <div key={i} aria-hidden="true" className="flex items-center gap-4 py-3 pl-3">
          <Skeleton circle className="h-2.5 w-2.5 shrink-0" />
          <Skeleton className="h-4 flex-1" />
          <Skeleton className="h-4 w-10 shrink-0" />
        </div>
      ))}
    </div>
  );
}
