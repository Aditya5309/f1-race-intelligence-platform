"use client";

import { GrowBar } from "@/components/motion";

/**
 * Horizontal progress-bar fill for win probability — the Pit Wall's
 * championship-points-bar pattern, applied to probability instead of
 * points. Track = hairline color, fill = the driver's constructor color,
 * width = probability (0-1).
 */
export function ProbabilityBar({
  probability,
  color,
  delay = 0,
}: {
  probability: number;
  color: string;
  delay?: number;
}) {
  const pct = Math.min(Math.max(probability, 0), 1) * 100;
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--color-hairline)]">
      <GrowBar percent={pct} color={color} delay={delay} className="h-full rounded-full" />
    </div>
  );
}
