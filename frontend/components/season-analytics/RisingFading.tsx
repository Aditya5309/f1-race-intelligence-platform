import type { DriverTrendRow } from "@/lib/season-aggregation";

/** delta is negative for an improving predicted rank (lower = better). */
export function RisingFading({ trend }: { trend: DriverTrendRow[] }) {
  if (!trend.length) {
    return (
      <p className="text-sm text-[var(--color-muted-text)]">
        Not enough multi-race history yet this season for a trend.
      </p>
    );
  }
  const rising = [...trend].sort((a, b) => a.delta - b.delta).slice(0, 3);
  const fading = [...trend].sort((a, b) => b.delta - a.delta).slice(0, 3);

  return (
    <div className="grid gap-6 sm:grid-cols-2">
      <div className="flex flex-col gap-2">
        <p className="label-caps text-xs text-[var(--color-muted-text)]">Rising</p>
        {rising.map((r) => (
          <div
            key={r.driver_id}
            className="flex items-center justify-between rounded-lg border border-[var(--color-hairline)] px-3 py-2"
          >
            <span className="text-sm">{r.label}</span>
            <span className="tabular text-sm">
              {r.recent_avg.toFixed(1)} avg rank{" "}
              <span className="text-[var(--color-live-green)]">{r.delta.toFixed(1)}</span>
            </span>
          </div>
        ))}
      </div>
      <div className="flex flex-col gap-2">
        <p className="label-caps text-xs text-[var(--color-muted-text)]">Fading</p>
        {fading.map((r) => (
          <div
            key={r.driver_id}
            className="flex items-center justify-between rounded-lg border border-[var(--color-hairline)] px-3 py-2"
          >
            <span className="text-sm">{r.label}</span>
            <span className="tabular text-sm">
              {r.recent_avg.toFixed(1)} avg rank{" "}
              <span className="text-[var(--color-accent-red)]">+{r.delta.toFixed(1)}</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
