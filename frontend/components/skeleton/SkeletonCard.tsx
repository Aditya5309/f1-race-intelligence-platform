import { Skeleton } from "./Skeleton";

const LINE_WIDTHS = ["w-3/4", "w-1/2", "w-5/6", "w-2/3", "w-1/3"];

/**
 * Generic rounded card placeholder. `variant="light"` matches the cream
 * secondary-card shape (EntityProfileCard, ModelComparisonTable's wrapper —
 * `bg-white/40`); `variant="dark"` matches the near-black feature-card
 * shape (ModelStatusCard, ModelCard — `bg-[var(--color-feature-card-bg)]`).
 * Both real families share the same `rounded-2xl p-6`, so only the fill
 * changes between variants.
 */
export function SkeletonCard({
  lines = 3,
  variant = "light",
  label = "Loading…",
  className,
}: {
  lines?: number;
  variant?: "light" | "dark";
  label?: string;
  className?: string;
}) {
  return (
    <div
      role="status"
      aria-live="polite"
      className={`rounded-2xl p-6 ${variant === "dark" ? "bg-[var(--color-feature-card-bg)]" : "bg-white/40"} ${className ?? ""}`}
    >
      <span className="sr-only">{label}</span>
      <div className="flex flex-col gap-3">
        {Array.from({ length: lines }).map((_, i) => (
          <Skeleton
            key={i}
            dark={variant === "dark"}
            className={`h-4 ${LINE_WIDTHS[i % LINE_WIDTHS.length]}`}
          />
        ))}
      </div>
    </div>
  );
}
