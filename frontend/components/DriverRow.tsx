"use client";

import { FadeIn } from "@/components/motion";
import { motion as motionTokens } from "@/lib/design-tokens";
import { constructorColor } from "@/lib/constructor-colors";
import type { DriverPrediction } from "@/lib/api";

/**
 * One ranked driver row: left-border accent bar + dark-ringed color dot
 * (the Phase 1 fix for pale team colors like Haas gray washing out on the
 * cream page), tabular win probability. Shared by Home's PredictionList
 * and Race Center's FullFieldTable — do not fork this markup per caller.
 */
export function DriverRow({
  driver,
  index,
  isWinner,
}: {
  driver: DriverPrediction;
  index: number;
  isWinner: boolean;
}) {
  return (
    <FadeIn
      as="li"
      x={-8}
      y={0}
      duration={motionTokens.numberTick.duration}
      ease={motionTokens.numberTick.ease}
      delay={index * motionTokens.staggerStep}
      className="flex items-center gap-4 py-3 pl-3"
      style={{ borderLeft: `4px solid ${constructorColor(driver.constructor_name)}` }}
    >
      <span className="tabular w-6 text-sm text-[var(--color-muted-text)]">
        {driver.predicted_rank}
      </span>
      <span
        className="h-2.5 w-2.5 shrink-0 rounded-full ring-1 ring-inset ring-black/20"
        style={{ backgroundColor: constructorColor(driver.constructor_name) }}
        aria-hidden
      />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium">
          {driver.driver_name ?? `Driver ${driver.driver_id}`}
          {isWinner && (
            <span className="label-caps ml-2 text-[10px] text-[var(--color-accent-red)]">
              Winner
            </span>
          )}
        </p>
        <p className="truncate text-xs text-[var(--color-muted-text)]">
          {driver.constructor_name ?? "—"}
        </p>
      </div>
      <span className="tabular text-sm">{(driver.win_probability * 100).toFixed(1)}%</span>
    </FadeIn>
  );
}
