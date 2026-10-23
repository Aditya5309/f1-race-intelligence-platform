"use client";

import { animate, motion, useMotionValue, useReducedMotion, useTransform } from "framer-motion";
import { useEffect } from "react";
import { motion as motionTokens } from "@/lib/design-tokens";

/**
 * Ticks a number from 0 up to `value` on mount/value change. The
 * DISPLAYED value during the tween is always an interpolation of the
 * exact `value` prop — never a separate, independently-rounded number —
 * so the final rendered text is always byte-identical to
 * `format(value)`, matching whatever the caller would have rendered
 * statically. Renders via a MotionValue subscription (not React state)
 * so the tick doesn't re-render the component every frame.
 */
export function AnimatedNumber({
  value,
  format = (n: number) => n.toFixed(0),
  duration = motionTokens.numberTick.duration,
  className,
}: {
  value: number;
  format?: (n: number) => string;
  duration?: number;
  className?: string;
}) {
  const prefersReducedMotion = useReducedMotion();
  const motionValue = useMotionValue(prefersReducedMotion ? value : 0);
  const display = useTransform(motionValue, (v) => format(v));

  useEffect(() => {
    if (prefersReducedMotion) {
      motionValue.set(value);
      return;
    }
    const controls = animate(motionValue, value, { duration, ease: "easeOut" });
    return () => controls.stop();
  }, [value, prefersReducedMotion, duration, motionValue]);

  return <motion.span className={className}>{display}</motion.span>;
}
