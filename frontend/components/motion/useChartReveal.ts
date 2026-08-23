"use client";

import { useRef, type RefObject } from "react";
import { useInView, useReducedMotion } from "framer-motion";

/**
 * Recharts has its own internal animation system, entirely separate from
 * Framer Motion — app/layout.tsx's `<MotionConfig reducedMotion="user">`
 * has zero effect on it. Every chart's `isAnimationActive` currently
 * defaults to `true` regardless of the OS setting; this hook is the one
 * shared place that reads the real preference so no chart component has
 * to duplicate that check. Only touches `isAnimationActive` — Recharts'
 * own default duration/easing are left exactly as they were (only
 * suppressed, never re-tuned) so normal-motion behavior is unchanged.
 *
 * `viewport: true` additionally defers rendering the chart until its
 * `ref` scrolls into view (see frontend-animation-visibility-audit.md) —
 * Recharts has no server-rendered fallback and animates on mount, so a
 * below-the-fold chart otherwise finishes its draw-in animation before
 * the user ever scrolls to it. `useInView` is called unconditionally
 * (Rules of Hooks) but its result is only honored when `viewport` is
 * requested; every existing zero-arg call site is unaffected — it keeps
 * `shouldRender: true` always, identical to its pre-existing behavior.
 * Orthogonal to reduced motion: gating controls *when* the chart mounts,
 * `isAnimationActive` (unchanged above) controls whether the mount is
 * eased or immediate, exactly like every other primitive in this file.
 *
 * Return shape is deliberately conditional, not just conditionally-valued:
 * every pre-existing call site does `const chartReveal = useChartReveal();
 * ... {...chartReveal}`, spreading the whole object straight onto a
 * Recharts series element. `ref` is a reserved React prop name — spreading
 * one into `{...chartReveal}` would silently attach it as that element's
 * ref. Omitting the `ref`/`shouldRender` keys entirely (not just setting
 * them to `undefined`) when `viewport` isn't requested keeps every
 * existing `{...chartReveal}` spread byte-identical to before this change.
 *
 * Overloaded (not just unioned) so a call site passing the `viewport: true`
 * literal gets the wider return type back without needing to narrow it
 * itself — every viewport-mode call site destructures `ref`/`shouldRender`
 * directly.
 */
export function useChartReveal(options: { viewport: true }): {
  isAnimationActive: boolean;
  ref: RefObject<HTMLDivElement | null>;
  shouldRender: boolean;
};
export function useChartReveal(options?: { viewport?: false }): {
  isAnimationActive: boolean;
};
export function useChartReveal(options: { viewport?: boolean } = {}):
  | { isAnimationActive: boolean }
  | { isAnimationActive: boolean; ref: RefObject<HTMLDivElement | null>; shouldRender: boolean } {
  const prefersReducedMotion = useReducedMotion();
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, amount: 0.2 });
  const isAnimationActive = !prefersReducedMotion;
  if (!options.viewport) return { isAnimationActive };
  return { isAnimationActive, ref, shouldRender: isInView };
}
