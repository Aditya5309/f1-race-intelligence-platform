"use client";

import { useEffect, useState } from "react";
import { simulateGrid, type DriverPrediction, type SimulateGridResponse } from "@/lib/api";

/**
 * Prediction Simulator — functional, deliberately plain styling this pass
 * (native range input, no custom slider skin, no chart). Motion/interaction
 * polish is a follow-up phase, per the Phase 2 plan.
 */
export function GridSimulator({
  raceId,
  predictions,
}: {
  raceId: number;
  predictions: DriverPrediction[];
}) {
  const [driverId, setDriverId] = useState(predictions[0]?.driver_id ?? 0);
  const [gridPosition, setGridPosition] = useState(1);
  const [pitLane, setPitLane] = useState(false);
  const [result, setResult] = useState<SimulateGridResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fieldSize = predictions.length;

  useEffect(() => {
    let cancelled = false;
    simulateGrid(
      raceId,
      driverId,
      pitLane ? { pitLane: true } : { gridPosition },
    )
      .then((r) => {
        if (!cancelled) {
          setResult(r);
          setError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [raceId, driverId, gridPosition, pitLane]);

  return (
    <div>
      <h2 className="font-[family-name:var(--font-display)] text-2xl font-semibold">
        What if? Grid position simulator
      </h2>
      <p className="mt-1 text-xs text-[var(--color-muted-text)]">
        Move a driver&apos;s starting grid slot — everything else (form, circuit
        history, real qualifying pace) stays locked to what actually happened.
      </p>

      <div className="mt-4 flex flex-wrap items-center gap-4">
        <select
          value={driverId}
          onChange={(e) => setDriverId(Number(e.target.value))}
          className="rounded border border-[var(--color-hairline)] bg-transparent px-2 py-1 text-sm"
        >
          {predictions.map((p) => (
            <option key={p.driver_id} value={p.driver_id}>
              {p.driver_name ?? `Driver ${p.driver_id}`}
            </option>
          ))}
        </select>

        <label className="flex items-center gap-2 text-sm">
          <input
            type="range"
            min={1}
            max={fieldSize}
            value={gridPosition}
            disabled={pitLane}
            onChange={(e) => setGridPosition(Number(e.target.value))}
          />
          <span className="tabular w-16">
            {pitLane ? "Pit lane" : `P${gridPosition}`}
          </span>
        </label>

        <label className="flex items-center gap-1 text-xs text-[var(--color-muted-text)]">
          <input
            type="checkbox"
            checked={pitLane}
            onChange={(e) => setPitLane(e.target.checked)}
          />
          Pit lane start
        </label>
      </div>

      {error && <p className="mt-3 text-xs text-[var(--color-accent-red)]">{error}</p>}

      {result && !error && (
        <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
          <div>
            <p className="label-caps text-[10px] text-[var(--color-muted-text)]">Real win share</p>
            <p className="tabular text-xl">{(result.real_win_probability * 100).toFixed(1)}%</p>
          </div>
          <div>
            <p className="label-caps text-[10px] text-[var(--color-muted-text)]">Simulated</p>
            <p className="tabular text-xl">
              {(result.simulated_win_probability * 100).toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="label-caps text-[10px] text-[var(--color-muted-text)]">Real grid</p>
            <p className="tabular text-xl">
              {result.real_grid_position != null ? `P${result.real_grid_position}` : "—"}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
