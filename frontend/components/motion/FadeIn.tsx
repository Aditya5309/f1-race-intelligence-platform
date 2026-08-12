"use client";

import { motion } from "framer-motion";
import type { CSSProperties, ReactNode } from "react";
import { motion as motionTokens } from "@/lib/design-tokens";

/**
 * Shared entrance fade — opacity 0->1 plus a small directional offset.
 * Defaults (y=12, motionTokens.cardEnter) match this app's original
 * card-entrance convention (HeroRaceCard/ModelStatusCard/RaceCenterHero);
 * override x/y/duration/ease for call sites that used a different offset
 * or timing token (e.g. DriverRow's x-offset + numberTick timing).
 */
export function FadeIn({
  children,
  as = "div",
  x = 0,
  y = 12,
  delay = 0,
  duration = motionTokens.cardEnter.duration,
  ease = motionTokens.cardEnter.ease,
  className,
  style,
}: {
  children: ReactNode;
  as?: "div" | "li";
  x?: number;
  y?: number;
  delay?: number;
  duration?: number;
  ease?: (typeof motionTokens.cardEnter)["ease"] | (typeof motionTokens.numberTick)["ease"];
  className?: string;
  style?: CSSProperties;
}) {
  const MotionTag = as === "li" ? motion.li : motion.div;
  return (
    <MotionTag
      initial={{ opacity: 0, x, y }}
      animate={{ opacity: 1, x: 0, y: 0 }}
      transition={{ duration, ease, delay }}
      className={className}
      style={style}
    >
      {children}
    </MotionTag>
  );
}
