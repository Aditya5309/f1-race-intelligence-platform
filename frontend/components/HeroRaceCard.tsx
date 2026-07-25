"use client";

import { motion } from "framer-motion";
import { motion as motionTokens } from "@/lib/design-tokens";
import type { RaceSummary, UpcomingRace } from "@/lib/api";

type Props =
  | { kind: "upcoming"; race: UpcomingRace }
  | { kind: "latest"; race: RaceSummary }
  | { kind: "none" };

export function HeroRaceCard(props: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={motionTokens.cardEnter}
    >
      {props.kind === "upcoming" && (
        <>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">
            Up next &middot; Round {props.race.round}, {props.race.year}
          </p>
          <h1 className="mt-2 font-[family-name:var(--font-display)] text-4xl font-semibold sm:text-5xl">
            <em className="italic text-[var(--color-accent-red)]">{props.race.name}</em>
          </h1>
          <p className="tabular mt-3 text-sm text-[var(--color-muted-text)]">
            {new Date(props.race.date).toLocaleDateString("en-US", {
              year: "numeric",
              month: "long",
              day: "numeric",
            })}
          </p>
        </>
      )}

      {props.kind === "latest" && (
        <>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">
            Latest scored race
          </p>
          <h1 className="mt-2 font-[family-name:var(--font-display)] text-4xl font-semibold sm:text-5xl">
            Round <em className="italic text-[var(--color-accent-red)]">{props.race.round}</em>,{" "}
            {props.race.year}
          </h1>
          <p className="tabular mt-3 text-sm text-[var(--color-muted-text)]">
            {props.race.n_drivers} drivers scored
          </p>
        </>
      )}

      {props.kind === "none" && (
        <h1 className="font-[family-name:var(--font-display)] text-4xl font-semibold">
          No races available
        </h1>
      )}
    </motion.div>
  );
}
