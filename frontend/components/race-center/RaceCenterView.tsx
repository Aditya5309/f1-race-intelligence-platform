"use client";

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import {
  getPredictions,
  predictUpcoming,
  type RaceSummary,
  type UpcomingRace,
} from "@/lib/api";
import { SkeletonHero, SkeletonList, SkeletonTable } from "@/components/skeleton";
import { RacePicker } from "./RacePicker";
import { RaceCenterHero } from "./RaceCenterHero";
import { PodiumRow } from "./PodiumRow";
import { ContenderCard } from "./ContenderCard";
import { FullFieldTable } from "./FullFieldTable";
import { GridSimulator } from "./GridSimulator";
import { HistoricalReplay } from "./HistoricalReplay";
import { QualifyingImpact } from "./QualifyingImpact";
import type { RaceCenterPrediction } from "./types";

export function RaceCenterView({
  races,
  upcoming,
  initialRaceId,
}: {
  races: RaceSummary[];
  upcoming: UpcomingRace | null;
  initialRaceId: number;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const [raceId, setRaceId] = useState(initialRaceId);
  const [data, setData] = useState<RaceCenterPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const isUpcoming = upcoming != null && raceId === upcoming.race_id;

  useEffect(() => {
    let cancelled = false;
    Promise.resolve().then(() => !cancelled && setIsLoading(true));

    const req = isUpcoming
      ? predictUpcoming(upcoming!.year, upcoming!.round).then((r) => ({
          race_id: upcoming!.race_id,
          year: r.year,
          round: r.round,
          prediction_id: r.prediction_id,
          generated_at: r.generated_at,
          model: r.model,
          predictions: r.predictions,
          actual_winner_driver_id: null,
          model_top1_hit: null,
          isUpcoming: true as const,
          materialization_status: r.materialization_status,
          missing_inputs: r.missing_inputs,
          caveats: r.caveats,
          provenance: r.provenance,
        }))
      : getPredictions(raceId).then((r) => ({
          race_id: r.race_id,
          year: r.year,
          round: r.round,
          prediction_id: r.prediction_id,
          generated_at: r.generated_at,
          model: r.model,
          predictions: r.predictions,
          actual_winner_driver_id: r.actual_winner_driver_id,
          model_top1_hit: r.model_top1_hit,
          isUpcoming: false as const,
        }));

    req
      .then((normalized) => {
        if (!cancelled) {
          setData(normalized);
          setError(null);
        }
      })
      .catch((err) => {
        // Stale data from the previous race is left in place rather than
        // cleared — an interim "old content" beat, not a broken page.
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => !cancelled && setIsLoading(false));

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raceId, isUpcoming]);

  function handleSelect(id: number) {
    setRaceId(id);
    router.replace(`${pathname}?race_id=${id}`, { scroll: false });
  }

  return (
    <div className="flex flex-col gap-8">
      <RacePicker races={races} upcoming={upcoming} selectedRaceId={raceId} onSelect={handleSelect} />

      {error && (
        <p className="text-sm text-[var(--color-accent-red)]">Couldn&apos;t load this race: {error}</p>
      )}

      {isLoading ? (
        <>
          <SkeletonHero variant="feature" label="Loading race prediction…" />
          <SkeletonList variant="grid" items={3} label="Loading predicted podium…" />
          <SkeletonTable rows={10} columns={4} hasHeader={false} label="Loading full field…" />
        </>
      ) : (
        data && (
          <>
            <RaceCenterHero data={data} />
            <PodiumRow predictions={data.predictions} winnerId={data.actual_winner_driver_id} />

            <div>
              <p className="label-caps text-xs text-[var(--color-muted-text)]">Top contenders</p>
              <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
                {data.predictions.slice(0, 5).map((driver, i) => (
                  <ContenderCard key={driver.driver_id} driver={driver} index={i} />
                ))}
              </div>
            </div>

            <FullFieldTable predictions={data.predictions} winnerId={data.actual_winner_driver_id} />

            {!data.isUpcoming && <HistoricalReplay data={data} />}
            {!data.isUpcoming && <QualifyingImpact raceId={data.race_id} />}
            {!data.isUpcoming && (
              <GridSimulator raceId={data.race_id} predictions={data.predictions} />
            )}
            {data.isUpcoming && (
              <p className="text-xs text-[var(--color-muted-text)]">
                Qualifying Impact and the Grid Simulator need a completed race to compare
                against — check back after this one runs.
              </p>
            )}
          </>
        )
      )}
    </div>
  );
}
