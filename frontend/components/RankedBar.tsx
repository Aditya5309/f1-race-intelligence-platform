"use client";

import { GrowBar } from "@/components/motion";
import { motion as motionTokens } from "@/lib/design-tokens";

/**
 * Horizontal bar generalized from race-center/ProbabilityBar.tsx to any
 * ranked value (points, pick counts) rather than a fixed 0-1 probability —
 * width = value / maxValue in the list, not an absolute fraction. Shared
 * across Season Analytics and Circuit Explorer.
 */
export function RankedBar({
  label,
  value,
  maxValue,
  displayValue,
  color,
  index = 0,
}: {
  label: string;
  value: number;
  maxValue: number;
  displayValue: string;
  color: string;
  index?: number;
}) {
  const pct = maxValue > 0 ? Math.min(value / maxValue, 1) * 100 : 0;
  return (
    <div className="flex items-center gap-3">
      <p className="w-28 shrink-0 truncate text-sm sm:w-36">{label}</p>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-[var(--color-hairline)]">
        <GrowBar
          percent={pct}
          color={color}
          delay={index * motionTokens.staggerStep}
          className="h-full rounded-full"
        />
      </div>
      <p className="tabular w-16 shrink-0 text-right text-xs text-[var(--color-muted-text)]">
        {displayValue}
      </p>
    </div>
  );
}
