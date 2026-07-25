"use client";

import type { RaceCenterPrediction } from "./types";

/**
 * Historical Replay — functional, plain styling this pass (no animated
 * reveal, per the Phase 2 plan). Only predicted-favorite vs. actual-winner
 * name + hit/miss: finish position and DNF status for a MISS aren't
 * available over HTTP (see the Phase 2 plan's data-availability note), so
 * this states the miss plainly rather than fabricating detail.
 */
export function HistoricalReplay({ data }: { data: RaceCenterPrediction }) {
  if (data.actual_winner_driver_id == null) return null;

  const top = data.predictions[0];
  const actualWinner = data.predictions.find(
    (p) => p.driver_id === data.actual_winner_driver_id,
  );

  return (
    <div>
      <h2 className="font-[family-name:var(--font-display)] text-2xl font-semibold">
        Historical replay
      </h2>
      <div className="mt-3 space-y-1 text-sm">
        <p>
          <span className="label-caps text-xs text-[var(--color-muted-text)]">Predicted:</span>{" "}
          {top?.driver_name ?? "—"} — {((top?.win_probability ?? 0) * 100).toFixed(1)}% win share
        </p>
        <p>
          <span className="label-caps text-xs text-[var(--color-muted-text)]">Actual:</span>{" "}
          {actualWinner?.driver_name ?? `Driver ${data.actual_winner_driver_id}`} won
        </p>
        <p className="text-xs text-[var(--color-muted-text)]">
          {data.model_top1_hit
            ? "The model's favorite won as predicted."
            : "The model's favorite did not win — see Full field above for who did."}
        </p>
      </div>
    </div>
  );
}
