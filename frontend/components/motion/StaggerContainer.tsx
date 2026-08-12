"use client";

import { motion } from "framer-motion";
import type { ReactNode } from "react";
import { motion as motionTokens } from "@/lib/design-tokens";

/**
 * Wraps a group of `StaggerItem`s — Framer Motion's native variants
 * propagation staggers each child's entrance by `staggerStep`, replacing
 * the previous manual `delay: index * staggerStep` computation with the
 * same resulting timing.
 */
export function StaggerContainer({
  children,
  staggerStep = motionTokens.staggerStep,
  className,
}: {
  children: ReactNode;
  staggerStep?: number;
  className?: string;
}) {
  return (
    <motion.div
      initial="hidden"
      animate="show"
      variants={{ hidden: {}, show: { transition: { staggerChildren: staggerStep } } }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
