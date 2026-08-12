"use client";

import { useReducedMotion } from "framer-motion";

/**
 * Recharts has its own internal animation system, entirely separate from
 * Framer Motion — app/layout.tsx's `<MotionConfig reducedMotion="user">`
 * has zero effect on it. Every chart's `isAnimationActive` currently
 * defaults to `true` regardless of the OS setting; this hook is the one
 * shared place that reads the real preference so no chart component has
 * to duplicate that check. Only touches `isAnimationActive` — Recharts'
 * own default duration/easing are left exactly as they were (only
 * suppressed, never re-tuned) so normal-motion behavior is unchanged.
 */
export function useChartReveal(): { isAnimationActive: boolean } {
  const prefersReducedMotion = useReducedMotion();
  return { isAnimationActive: !prefersReducedMotion };
}
