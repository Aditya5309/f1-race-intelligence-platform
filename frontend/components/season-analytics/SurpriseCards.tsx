import type { SeasonPredictionRow } from "@/lib/season-aggregation";

/**
 * Race names aren't reachable over HTTP for historical races (the same
 * metadata.py gap Phase 2 hit for Race Center) — "Round N" is the
 * established fallback (Decision 059), reused rather than reinvented.
 */
export function SurpriseCards({ rows }: { rows: SeasonPredictionRow[] }) {
  if (!rows.length) {
    return (
      <p className="text-sm text-[var(--color-muted-text)]">
        No decided races in this season&apos;s data.
      </p>
    );
  }
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      {rows.map((r) => (
        <div key={r.race_id} className="rounded-xl bg-white/40 p-4">
          <p className="text-sm font-medium">Round {r.round}</p>
          <p className="mt-1 text-xs text-[var(--color-muted-text)]">
            Won by {r.driver_name ?? `driver ${r.driver_id}`}
          </p>
          <p className="mt-1 text-xs text-[var(--color-muted-text)]">
            Model rank #{r.predicted_rank} &middot; {(r.win_probability * 100).toFixed(1)}% win share
          </p>
        </div>
      ))}
    </div>
  );
}
