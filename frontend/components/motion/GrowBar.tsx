"use client";

import { motion } from "framer-motion";
import { motion as motionTokens } from "@/lib/design-tokens";

/**
 * A single bar-fill growth animation (width 0 -> value%) — the shared
 * shape behind RankedBar's/ProbabilityBar's inline bar. Percent is
 * computed by the caller (clamping/max-value logic stays call-site
 * specific); this only owns the width tween.
 *
 * `viewport` (default false — mount-triggered, unchanged): see FadeIn's
 * doc comment for the full rationale. Needed independently of any
 * viewport-gated FadeIn it happens to be nested inside — a wrapper's
 * `whileInView` only gates the wrapper's own animated values, not a
 * child's separate `animate` call, so a bar nested in a viewport-gated
 * card must opt into `viewport` itself or it will still grow off-screen.
 */
export function GrowBar({
  percent,
  color,
  delay = 0,
  duration = motionTokens.numberTick.duration,
  ease = motionTokens.numberTick.ease,
  viewport = false,
  className,
}: {
  percent: number;
  color: string;
  delay?: number;
  duration?: number;
  ease?: (typeof motionTokens.numberTick)["ease"];
  viewport?: boolean;
  className?: string;
}) {
  const trigger = viewport
    ? { whileInView: { width: `${percent}%` }, viewport: { once: true, amount: 0.2 } }
    : { animate: { width: `${percent}%` } };
  return (
    <motion.div
      className={className}
      style={{ backgroundColor: color }}
      initial={{ width: 0 }}
      {...trigger}
      transition={{ duration, ease, delay }}
    />
  );
}
