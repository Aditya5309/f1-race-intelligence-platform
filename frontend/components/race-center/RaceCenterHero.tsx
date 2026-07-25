"use client";

import { motion } from "framer-motion";
import { motion as motionTokens } from "@/lib/design-tokens";
import { confidenceTier } from "@/lib/confidence";
import type { RaceCenterPrediction } from "./types";

/** Small CSS-only checkered corner motif — a restrained motorsport accent, not a background fill. */
function CheckeredCorner() {
  const cell = "10px";
  return (
    <div
      aria-hidden
      className="pointer-events-none absolute top-0 right-0 h-10 w-10 opacity-20"
      style={{
        backgroundImage: `
          linear-gradient(45deg, var(--color-text-on-dark) 25%, transparent 25%),
          linear-gradient(-45deg, var(--color-text-on-dark) 25%, transparent 25%),
          linear-gradient(45deg, transparent 75%, var(--color-text-on-dark) 75%),
          linear-gradient(-45deg, transparent 75%, var(--color-text-on-dark) 75%)`,
        backgroundSize: `${cell} ${cell}`,
        backgroundPosition: `0 0, 0 5px, 5px -5px, -5px 0px`,
      }}
    />
  );
}

export function RaceCenterHero({ data }: { data: RaceCenterPrediction }) {
  const top = data.predictions[0];
  const { label, stars } = confidenceTier(top?.win_probability ?? 0);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={motionTokens.cardEnter}
      className="relative overflow-hidden rounded-2xl bg-[var(--color-feature-card-bg)] p-8 text-[var(--color-text-on-dark)]"
    >
      <CheckeredCorner />
      <p className="label-caps text-xs text-[var(--color-muted-text)]">
        Round {data.round} · {data.year} {data.isUpcoming && "· Up next"}
      </p>
      <h1 className="mt-2 font-[family-name:var(--font-display)] text-3xl font-semibold sm:text-4xl">
        Predicted{" "}
        <em className="italic text-[var(--color-accent-red)]">
          {top?.driver_name ?? "winner"}
        </em>
      </h1>
      {top?.constructor_name && (
        <p className="mt-1 text-sm text-[var(--color-muted-text)]">{top.constructor_name}</p>
      )}

      <div className="mt-6 flex items-end gap-6">
        <span className="tabular text-6xl font-semibold leading-none">
          {((top?.win_probability ?? 0) * 100).toFixed(1)}
          <span className="text-2xl">%</span>
        </span>
        <div className="pb-1">
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Win share</p>
          <p className="mt-1 text-sm">
            <span aria-hidden>{"★".repeat(stars) + "☆".repeat(5 - stars)}</span>{" "}
            <span className="text-[var(--color-muted-text)]">{label}</span>
          </p>
        </div>
      </div>

      {data.isUpcoming && data.materialization_status && (
        <p className="mt-4 text-xs text-[var(--color-muted-text)]">
          {data.materialization_status === "pre_qualifying"
            ? "⚠ Provisional — qualifying not yet complete for this race."
            : "🔮 Pre-race prediction — the grid is set, this race hasn't run yet."}
        </p>
      )}
      {!data.isUpcoming && data.model_top1_hit !== null && (
        <p className="mt-4 text-xs text-[var(--color-muted-text)]">
          {data.model_top1_hit ? "✅ The model called it." : "❌ The model missed the winner."}
        </p>
      )}
    </motion.div>
  );
}
