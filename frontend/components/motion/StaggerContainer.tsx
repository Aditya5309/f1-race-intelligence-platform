"use client";

import { motion } from "framer-motion";
import type { ReactNode } from "react";
import { motion as motionTokens } from "@/lib/design-tokens";

/**
 * Wraps a group of `StaggerItem`s — Framer Motion's native variants
 * propagation staggers each child's entrance by `staggerStep`, replacing
 * the previous manual `delay: index * staggerStep` computation with the
 * same resulting timing.
 *
 * `viewport` (default false — mount-triggered, unchanged): see FadeIn's
 * doc comment. `StaggerItem`'s own `hidden`/`show` variants apply either
 * way — it just inherits whichever trigger this container uses.
 */
export function StaggerContainer({
  children,
  staggerStep = motionTokens.staggerStep,
  viewport = false,
  className,
}: {
  children: ReactNode;
  staggerStep?: number;
  viewport?: boolean;
  className?: string;
}) {
  const trigger = viewport
    ? { whileInView: "show", viewport: { once: true, amount: 0.2 } }
    : { animate: "show" };
  return (
    <motion.div
      initial="hidden"
      {...trigger}
      variants={{ hidden: {}, show: { transition: { staggerChildren: staggerStep } } }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
