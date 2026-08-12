import { NavShell } from "@/components/NavShell";
import { HeroRaceCard } from "@/components/HeroRaceCard";
import { ModelStatusCard } from "@/components/ModelStatusCard";
import { PredictionList } from "@/components/PredictionList";
import {
  getHealth,
  getUpcomingRace,
  getPredictions,
  listRaces,
  type HealthResponse,
  type PredictionResponse,
  type RaceSummary,
} from "@/lib/api";

/** Every field below traces to a live API call — nothing hardcoded, nothing mocked. */
export default async function Home() {
  const health: HealthResponse | null = await getHealth().catch(() => null);
  const upcoming = await getUpcomingRace().catch(() => null);
  const races: RaceSummary[] = await listRaces().catch(() => []);

  const latestRace = races.reduce<RaceSummary | null>((best, r) => {
    if (!best) return r;
    return r.year > best.year || (r.year === best.year && r.round > best.round) ? r : best;
  }, null);

  const prediction: PredictionResponse | null = latestRace
    ? await getPredictions(latestRace.race_id).catch(() => null)
    : null;

  return (
    <>
      <NavShell />
      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-10 px-6 py-12">
        <section className="grid gap-8 md:grid-cols-[2fr_1fr]">
          {upcoming ? (
            <HeroRaceCard kind="upcoming" race={upcoming} />
          ) : latestRace ? (
            <HeroRaceCard kind="latest" race={latestRace} />
          ) : (
            <HeroRaceCard kind="none" />
          )}
          <ModelStatusCard health={health} />
        </section>

        {prediction && (
          <section>
            {upcoming && (
              <div className="mb-3">
                <p className="label-caps text-xs text-[var(--color-muted-text)]">
                  Latest scored race
                </p>
                <p className="text-sm">
                  Round {latestRace!.round}, {latestRace!.year}
                </p>
              </div>
            )}
            <PredictionList prediction={prediction} />
          </section>
        )}
      </main>
      <footer className="border-t border-[var(--color-hairline)] px-6 py-6">
        <p className="label-caps text-center text-xs text-[var(--color-muted-text)]">
          F1 Race Winner Prediction &middot; Phase 1 pilot
        </p>
      </footer>
    </>
  );
}
