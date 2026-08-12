"use client";

import { motion } from "framer-motion";
import { motion as motionTokens } from "@/lib/design-tokens";

/**
 * A single bar-fill growth animation (width 0 -> value%) — the shared
 * shape behind RankedBar's/ProbabilityBar's inline bar. Percent is
 * computed by the caller (clamping/max-value logic stays call-site
 * specific); this only owns the width tween.
 */
export function GrowBar({
  percent,
  color,
  delay = 0,
  duration = motionTokens.numberTick.duration,
  ease = motionTokens.numberTick.ease,
  className,
}: {
  percent: number;
  color: string;
  delay?: number;
  duration?: number;
  ease?: (typeof motionTokens.numberTick)["ease"];
  className?: string;
}) {
  return (
    <motion.div
      className={className}
      style={{ backgroundColor: color }}
      initial={{ width: 0 }}
      animate={{ width: `${percent}%` }}
      transition={{ duration, ease, delay }}
    />
  );
}
