"use client";

import { motion } from "framer-motion";
import { motion as motionTokens } from "@/lib/design-tokens";
import { constructorColor } from "@/lib/constructor-colors";
import type { DriverPrediction } from "@/lib/api";

const MEDALS = ["🥇", "🥈", "🥉"];

export function PodiumRow({
  predictions,
  winnerId,
}: {
  predictions: DriverPrediction[];
  winnerId: number | null;
}) {
  const podium = predictions.slice(0, 3);
  return (
    <div>
      <p className="label-caps text-xs text-[var(--color-muted-text)]">Predicted podium</p>
      <div className="mt-3 grid gap-3 sm:grid-cols-3">
        {podium.map((driver, i) => (
          <motion.div
            key={driver.driver_id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ ...motionTokens.cardEnter, delay: i * motionTokens.staggerStep }}
            className="rounded-xl bg-white/40 p-4"
            style={{ borderLeft: `4px solid ${constructorColor(driver.constructor_name)}` }}
          >
            <p className="text-lg">
              {MEDALS[i]}{" "}
              <span className="text-sm font-medium">
                {driver.driver_name ?? `Driver ${driver.driver_id}`}
              </span>
            </p>
            <p className="truncate text-xs text-[var(--color-muted-text)]">
              {driver.constructor_name ?? "—"}
            </p>
            <p className="tabular mt-2 text-sm">{(driver.win_probability * 100).toFixed(1)}%</p>
            {winnerId != null && driver.driver_id === winnerId && (
              <span className="label-caps mt-1 inline-block text-[10px] text-[var(--color-accent-red)]">
                Actual winner
              </span>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  );
}
