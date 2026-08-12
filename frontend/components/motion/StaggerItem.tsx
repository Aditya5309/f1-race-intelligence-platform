"use client";

import { motion } from "framer-motion";
import type { CSSProperties, ReactNode } from "react";
import { motion as motionTokens } from "@/lib/design-tokens";

/**
 * One child of a `StaggerContainer` — must be a direct child so Framer
 * Motion's variants propagation applies the parent's `staggerChildren`
 * delay automatically. Same fade+offset shape as `FadeIn`, but timing is
 * driven by the parent, not an explicit `delay` prop.
 */
export function StaggerItem({
  children,
  x = 0,
  y = 12,
  duration = motionTokens.cardEnter.duration,
  ease = motionTokens.cardEnter.ease,
  className,
  style,
}: {
  children: ReactNode;
  x?: number;
  y?: number;
  duration?: number;
  ease?: (typeof motionTokens.cardEnter)["ease"];
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <motion.div
      variants={{
        hidden: { opacity: 0, x, y },
        show: { opacity: 1, x: 0, y: 0, transition: { duration, ease } },
      }}
      className={className}
      style={style}
    >
      {children}
    </motion.div>
  );
}
