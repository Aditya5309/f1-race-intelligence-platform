import { Skeleton } from "./Skeleton";

/** Matches EntityProfileCard's shell exactly: bordered secondary card,
 * avatar dot + stacked name/subtitle block. */
export function SkeletonEntityProfile({
  label = "Loading profile…",
}: {
  label?: string;
}) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="rounded-2xl border border-[var(--color-hairline)] bg-white/40 p-6"
    >
      <span className="sr-only">{label}</span>
      <div className="flex items-start gap-4">
        <Skeleton circle className="mt-1.5 h-4 w-4 shrink-0" />
        <div className="flex flex-1 flex-col gap-2">
          <Skeleton className="h-7 w-1/2" />
          <Skeleton className="h-4 w-1/3" />
        </div>
      </div>
    </div>
  );
}
