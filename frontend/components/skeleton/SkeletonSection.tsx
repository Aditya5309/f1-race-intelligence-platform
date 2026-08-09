import { Skeleton } from "./Skeleton";

/** Generic section wrapper: a label-caps placeholder + a children skeleton
 * block, matching the "label-caps text-xs" + "mt-3 content" pattern used
 * throughout this app (e.g. PodiumRow, RankedBar groups). The default
 * `labelWidth` is a fixed Tailwind width, a deliberate, approved override
 * of §1's general "no fixed pixel width" rule — see SkeletonHero.tsx's
 * comment for the reasoning (short label text doesn't scale with
 * container width, so a relative width would be less faithful). */
export function SkeletonSection({
  labelWidth = "w-24",
  label = "Loading…",
  children,
}: {
  labelWidth?: string;
  label?: string;
  children: React.ReactNode;
}) {
  return (
    <section role="status" aria-live="polite">
      <span className="sr-only">{label}</span>
      <Skeleton className={`h-3 ${labelWidth}`} />
      <div className="mt-3">{children}</div>
    </section>
  );
}
