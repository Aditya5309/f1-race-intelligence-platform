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
 *
 * `viewport` (frontend-animation-visibility-audit.md P0 fix): both lists
 * can run to 20+ rows, so unlike a short card grid, most of a long list
 * mounts below the fold — without this, those rows finished their FadeIn
 * before the user ever scrolled to them. Each row's own IntersectionObserver
 * fires independently as it individually scrolls into view. The stagger
 * `delay` below is capped (not removed) for the same reason: it was tuned
 * for rows that all mount at once (a "wave" down the initially-visible
 * batch); left uncapped under per-row viewport-triggering, a deep-list row
 * revealed by scrolling would wait an extra `index * staggerStep` (over a
 * second, by row 20+) *after* the user already scrolled it into view —
 * capping keeps the wave for the visible batch without that compounding,
 * disconnected-feeling lag for rows revealed later by scrolling.
 */
const MAX_STAGGER_INDEX = 6;

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
      delay={Math.min(index, MAX_STAGGER_INDEX) * motionTokens.staggerStep}
      viewport
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
