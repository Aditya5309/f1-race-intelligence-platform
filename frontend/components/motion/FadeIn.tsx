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
 *
 * `viewport` (default false — mount-triggered, unchanged) opts into
 * viewport-triggered entrance instead: `animate` becomes `whileInView`, so
 * the element stays at its `initial` state until it scrolls into view,
 * then plays the same transition once (see
 * frontend-animation-visibility-audit.md — apply only to call sites with a
 * confirmed below-the-fold finding, not speculatively). Reduced motion is
 * unaffected either way — it's handled entirely by app/layout.tsx's
 * `MotionConfig`, which suppresses the transition itself, not whether/when
 * it's triggered.
 */
export function FadeIn({
  children,
  as = "div",
  x = 0,
  y = 12,
  delay = 0,
  duration = motionTokens.cardEnter.duration,
  ease = motionTokens.cardEnter.ease,
  viewport = false,
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
  viewport?: boolean;
  className?: string;
  style?: CSSProperties;
}) {
  const MotionTag = as === "li" ? motion.li : motion.div;
  const trigger = viewport
    ? { whileInView: { opacity: 1, x: 0, y: 0 }, viewport: { once: true, amount: 0.2 } }
    : { animate: { opacity: 1, x: 0, y: 0 } };
  return (
    <MotionTag
      initial={{ opacity: 0, x, y }}
      {...trigger}
      transition={{ duration, ease, delay }}
      className={className}
      style={style}
    >
      {children}
    </MotionTag>
  );
}
