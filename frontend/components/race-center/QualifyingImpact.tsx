"use client";

import { useEffect, useState } from "react";
import { DriverRow } from "@/components/DriverRow";
import { FetchErrorMessage } from "@/components/FetchErrorMessage";
import { ApiError, getVsBaseline, type BaselineComparisonResponse } from "@/lib/api";

/** Model vs. pole-only baseline — functional, plain two-column layout this pass. */
export function QualifyingImpact({ raceId }: { raceId: number }) {
  const [data, setData] = useState<BaselineComparisonResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [baselineUnavailable, setBaselineUnavailable] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getVsBaseline(raceId)
      .then((r) => {
        if (!cancelled) {
          setData(r);
          setError(null);
          setBaselineUnavailable(false);
        }
      })
      .catch((err) => {
        if (cancelled) return;
        // The backend raises exactly 503 for "baseline model not available"
        // (app/api.py's predictions_vs_baseline) — that documented case
        // degrades silently. Everything else (404, 500, a raw network
        // failure that isn't even an ApiError) is a genuine error.
        if (err instanceof ApiError && err.status === 503) {
          setBaselineUnavailable(true);
          setError(null);
        } else {
          setError(err instanceof Error ? err.message : String(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [raceId]);

  if (baselineUnavailable) return null;
  if (error) return <FetchErrorMessage message={`Couldn't load qualifying impact: ${error}`} />;

  return (
    <div>
      <h2 className="font-[family-name:var(--font-display)] text-2xl font-semibold">
        Qualifying impact
      </h2>
      <p className="mt-1 text-xs text-[var(--color-muted-text)]">
        Full model vs. a pole-only heuristic — how much of the pick is really just qualifying.
      </p>

      {data && (
        <div className="mt-4 grid gap-6 sm:grid-cols-2">
          <div>
            <p className="label-caps text-xs text-[var(--color-muted-text)]">Full model</p>
            <ol className="mt-2 divide-y divide-[var(--color-hairline)]">
              {data.model_predictions.slice(0, 5).map((d, i) => (
                <DriverRow
                  key={d.driver_id}
                  driver={d}
                  index={i}
                  isWinner={data.actual_winner_driver_id === d.driver_id}
                />
              ))}
            </ol>
          </div>
          <div>
            <p className="label-caps text-xs text-[var(--color-muted-text)]">Pole-only baseline</p>
            <ol className="mt-2 divide-y divide-[var(--color-hairline)]">
              {data.baseline_predictions.slice(0, 5).map((d, i) => (
                <DriverRow
                  key={d.driver_id}
                  driver={d}
                  index={i}
                  isWinner={data.actual_winner_driver_id === d.driver_id}
                />
              ))}
            </ol>
          </div>
        </div>
      )}
    </div>
  );
}
