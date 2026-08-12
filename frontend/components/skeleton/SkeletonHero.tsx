import { Skeleton } from "./Skeleton";

/**
 * `variant="feature"` matches RaceCenterHero's near-black `rounded-2xl p-8`
 * card (label line, big heading, subtitle, a large win-share stat, a star
 * line). `variant="bare"` matches HeroRaceCard's actual Home-page layout —
 * naked stacked lines directly on the cream background, no card chrome at
 * all (label line, heading, date line) — these are two genuinely different
 * real shapes, not one fixed hero shape app-wide.
 *
 * Label/stat placeholder widths below use fixed Tailwind widths (w-40,
 * w-32, etc.), a deliberate, approved override of §1's general "no fixed
 * pixel width" rule: the real text these stand in for (round labels, stat
 * numbers) doesn't scale with container width either, so a relative width
 * would be less faithful, not more — do not "fix" these back to w-full/
 * fractional widths without re-confirming that reasoning still holds.
 */
export function SkeletonHero({
  variant = "feature",
  label = "Loading race summary…",
}: {
  variant?: "feature" | "bare";
  label?: string;
}) {
  if (variant === "bare") {
    return (
      <div role="status" aria-live="polite" className="flex flex-col gap-3">
        <span className="sr-only">{label}</span>
        <Skeleton className="h-3 w-48" />
        <Skeleton className="h-10 w-3/4" />
        <Skeleton className="h-4 w-1/3" />
      </div>
    );
  }

  return (
    <div
      role="status"
      aria-live="polite"
      className="rounded-2xl bg-[var(--color-feature-card-bg)] p-8"
    >
      <span className="sr-only">{label}</span>
      <Skeleton dark className="h-3 w-40" />
      <Skeleton dark className="mt-3 h-9 w-2/3" />
      <Skeleton dark className="mt-2 h-4 w-1/3" />
      <div className="mt-6 flex items-end gap-6">
        <Skeleton dark className="h-14 w-32" />
        <div className="flex flex-col gap-2 pb-1">
          <Skeleton dark className="h-3 w-16" />
          <Skeleton dark className="h-4 w-24" />
        </div>
      </div>
    </div>
  );
}
