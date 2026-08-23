"use client";

import { FadeIn } from "@/components/motion";
import { motion as motionTokens } from "@/lib/design-tokens";
import { constructorColor } from "@/lib/constructor-colors";
import { accessibleBadgeTextColor } from "@/lib/accessible-badge-color";
import { teamCode } from "@/lib/team-codes";
import { ProbabilityBar } from "./ProbabilityBar";
import type { DriverPrediction } from "@/lib/api";

/**
 * `viewport` (frontend-animation-visibility-audit.md P0 fix): this grid's
 * cards sit below RaceCenterHero + PodiumRow, confirmed below the fold on
 * initial load. The nested ProbabilityBar/GrowBar needs `viewport` passed
 * through too, independently of this card's own FadeIn — a parent's
 * `whileInView` only gates its own animated values, not a child's separate
 * `animate` call, so without this the bar would still grow to full width
 * off-screen the instant the (still node-invisible) card mounts.
 */
export function ContenderCard({ driver, index }: { driver: DriverPrediction; index: number }) {
  const color = constructorColor(driver.constructor_name);
  const badgeTextColor = accessibleBadgeTextColor(color);
  return (
    <FadeIn
      y={10}
      delay={index * motionTokens.staggerStep}
      viewport
      className="rounded-xl bg-white/40 p-4"
      style={{ borderLeft: `4px solid ${color}` }}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium">
            {driver.driver_name ?? `Driver ${driver.driver_id}`}
          </p>
          <p className="truncate text-xs text-[var(--color-muted-text)]">
            {driver.constructor_name ?? "—"}
          </p>
        </div>
        <span
          className="label-caps shrink-0 rounded-full px-2 py-0.5 text-[10px] ring-1 ring-inset ring-black/15"
          style={{ backgroundColor: `${color}22`, color: badgeTextColor }}
        >
          {teamCode(driver.constructor_name)}
        </span>
      </div>

      <p className="tabular mt-3 text-2xl font-semibold">
        {(driver.win_probability * 100).toFixed(1)}%
      </p>
      <p className="label-caps text-[10px] text-[var(--color-muted-text)]">Win share</p>

      <div className="mt-2">
        <ProbabilityBar
          probability={driver.win_probability}
          color={color}
          delay={index * motionTokens.staggerStep}
          viewport
        />
      </div>
    </FadeIn>
  );
}
