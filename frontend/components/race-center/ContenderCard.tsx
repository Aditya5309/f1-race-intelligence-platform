"use client";

import { motion } from "framer-motion";
import { motion as motionTokens } from "@/lib/design-tokens";
import { constructorColor } from "@/lib/constructor-colors";
import { teamCode } from "@/lib/team-codes";
import { ProbabilityBar } from "./ProbabilityBar";
import type { DriverPrediction } from "@/lib/api";

export function ContenderCard({ driver, index }: { driver: DriverPrediction; index: number }) {
  const color = constructorColor(driver.constructor_name);
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ ...motionTokens.cardEnter, delay: index * motionTokens.staggerStep }}
      className="rounded-xl bg-white/40 p-4"
      style={{ borderLeft: `4px solid ${color}` }}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium">
            {driver.driver_name ?? `Driver ${driver.driver_id}`}
          </p>
          <p className="truncate text-xs text-[var(--color-muted-text)]">
            {driver.constructor_name ?? "—"}
          </p>
        </div>
        <span
          className="label-caps shrink-0 rounded-full px-2 py-0.5 text-[10px] ring-1 ring-inset ring-black/15"
          style={{ backgroundColor: `${color}22`, color }}
        >
          {teamCode(driver.constructor_name)}
        </span>
      </div>

      <p className="tabular mt-3 text-2xl font-semibold">
        {(driver.win_probability * 100).toFixed(1)}%
      </p>
      <p className="label-caps text-[10px] text-[var(--color-muted-text)]">Win share</p>

      <div className="mt-2">
        <ProbabilityBar
          probability={driver.win_probability}
          color={color}
          delay={index * motionTokens.staggerStep}
        />
      </div>
    </motion.div>
  );
}
