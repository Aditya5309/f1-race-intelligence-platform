"use client";

import { useState } from "react";
import { DriverRow } from "@/components/DriverRow";
import type { DriverPrediction } from "@/lib/api";

/** Extends Phase 1's PredictionList left-border/dot row pattern — reused via
 * DriverRow, not reinvented — with the top10/rest split Streamlit's own
 * Race Center table uses. */
export function FullFieldTable({
  predictions,
  winnerId,
}: {
  predictions: DriverPrediction[];
  winnerId: number | null;
}) {
  const [expanded, setExpanded] = useState(false);
  const top10 = predictions.slice(0, 10);
  const rest = predictions.slice(10);

  return (
    <div>
      <h2 className="font-[family-name:var(--font-display)] text-2xl font-semibold">
        Full field
      </h2>
      <ol className="mt-4 divide-y divide-[var(--color-hairline)]">
        {top10.map((driver, i) => (
          <DriverRow
            key={driver.driver_id}
            driver={driver}
            index={i}
            isWinner={winnerId != null && driver.driver_id === winnerId}
          />
        ))}
      </ol>

      {rest.length > 0 && (
        <div className="mt-2">
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="label-caps text-xs text-[var(--color-accent-blue)]"
          >
            {expanded ? "Hide" : "Show"} rest of the field ({rest.length})
          </button>
          {expanded && (
            <ol className="mt-2 divide-y divide-[var(--color-hairline)]">
              {rest.map((driver, i) => (
                <DriverRow
                  key={driver.driver_id}
                  driver={driver}
                  index={i}
                  isWinner={winnerId != null && driver.driver_id === winnerId}
                />
              ))}
            </ol>
          )}
        </div>
      )}
    </div>
  );
}
