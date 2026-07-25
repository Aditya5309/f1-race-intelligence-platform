"use client";

import { DriverRow } from "@/components/DriverRow";
import type { PredictionResponse } from "@/lib/api";

export function PredictionList({ prediction }: { prediction: PredictionResponse }) {
  const winnerId = prediction.actual_winner_driver_id;
  const hasResult = winnerId != null;

  return (
    <div>
      <div className="flex items-baseline justify-between">
        <h2 className="font-[family-name:var(--font-display)] text-2xl font-semibold">
          Win probability
        </h2>
        {hasResult && (
          <span className="label-caps text-xs text-[var(--color-muted-text)]">
            {prediction.model_top1_hit ? "Model called it" : "Model missed"}
          </span>
        )}
      </div>

      <ol className="mt-4 divide-y divide-[var(--color-hairline)]">
        {prediction.predictions.map((driver, i) => (
          <DriverRow
            key={driver.driver_id}
            driver={driver}
            index={i}
            isWinner={hasResult && driver.driver_id === winnerId}
          />
        ))}
      </ol>
    </div>
  );
}
