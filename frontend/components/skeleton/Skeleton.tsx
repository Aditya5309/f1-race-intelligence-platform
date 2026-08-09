/**
 * Base shimmering block every other skeleton composes from. Purely
 * decorative — always `aria-hidden`, never a standalone loading
 * announcement (that's the shaped components' job, each of which wraps
 * one or more of these in its own `role="status"`).
 */
export function Skeleton({
  className,
  circle = false,
  dark = false,
}: {
  className?: string;
  circle?: boolean;
  dark?: boolean;
}) {
  return (
    <div
      aria-hidden="true"
      className={`${dark ? "skeleton-shimmer-dark" : "skeleton-shimmer"} ${circle ? "rounded-full" : "rounded-lg"} ${className ?? ""}`}
    />
  );
}
