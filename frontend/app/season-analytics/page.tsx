import { NavShell } from "@/components/NavShell";
import { StatRow } from "@/components/StatRow";
import { AccuracyChart } from "@/components/season-analytics/AccuracyChart";
import { ConstructorOutlook } from "@/components/season-analytics/ConstructorOutlook";
import { MostPredictedWinners } from "@/components/season-analytics/MostPredictedWinners";
import { RisingFading } from "@/components/season-analytics/RisingFading";
import { StandingsPanel } from "@/components/season-analytics/StandingsPanel";
import { SurpriseCards } from "@/components/season-analytics/SurpriseCards";
import { WinShareHistogram } from "@/components/season-analytics/WinShareHistogram";
import { YearPicker } from "@/components/season-analytics/YearPicker";
import { getStandings, listRaces, type RaceSummary } from "@/lib/api";
import {
  accuracySeries,
  avgFieldSize,
  byRace,
  constructorOutlook,
  driverTrend,
  getSeasonPredictions,
  hitRate,
  mostPredictedWinners,
  seasonInsights,
  surprises,
  topPicks,
  winShareHistogramBins,
} from "@/lib/season-aggregation";

export default async function SeasonAnalyticsPage({
  searchParams,
}: {
  searchParams: Promise<{ year?: string }>;
}) {
  const races: RaceSummary[] = await listRaces().catch(() => []);
  const years = Array.from(new Set(races.map((r) => r.year))).sort((a, b) => b - a);

  const params = await searchParams;
  const requestedYear = params.year ? Number(params.year) : undefined;
  const year = requestedYear && years.includes(requestedYear) ? requestedYear : years[0];

  if (year == null) {
    return (
      <>
        <NavShell />
        <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-4 px-6 py-12">
          <p className="text-sm text-[var(--color-muted-text)]">No races available from the API.</p>
        </main>
      </>
    );
  }

  const [rows, standings] = await Promise.all([
    getSeasonPredictions(year).catch(() => []),
    getStandings(year).catch(() => ({ year, drivers: [], constructors: [] })),
  ]);

  const header = (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div>
        <h1 className="font-[family-name:var(--font-display)] text-3xl font-semibold">
          Season Analytics
        </h1>
        <p className="mt-1 text-sm text-[var(--color-muted-text)]">
          How the {year} season is unfolding, round by round.
        </p>
      </div>
      <YearPicker years={years} selected={year} />
    </div>
  );

  if (!rows.length) {
    return (
      <>
        <NavShell />
        <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-4 px-6 py-12">
          {header}
          <p className="text-sm text-[var(--color-muted-text)]">No prediction data for {year}.</p>
        </main>
      </>
    );
  }

  const raceRows = byRace(rows);
  const rate = hitRate(raceRows);
  const picks = topPicks(rows);
  const surprisingRows = surprises(rows);
  const trend = driverTrend(rows);
  const insights = seasonInsights(picks, surprisingRows, trend);

  const teamByDriver = new Map<number, string | null>();
  for (const r of rows) teamByDriver.set(r.driver_id, r.constructor_name);

  return (
    <>
      <NavShell />
      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-10 px-6 py-12">
        {header}

        <StatRow
          stats={[
            { label: "Races", value: String(raceRows.length) },
            {
              label: "Model hit rate",
              value: rate != null ? `${(rate * 100).toFixed(0)}%` : "—",
              help: "Share of races where the model's #1 pick actually won",
            },
            { label: "Avg field size", value: avgFieldSize(rows).toFixed(0) },
          ]}
        />

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Insights</p>
          {insights.length ? (
            <ul className="mt-2 flex flex-col gap-1 text-sm text-[var(--color-muted-text)]">
              {insights.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-sm text-[var(--color-muted-text)]">
              Not enough decided races yet this season for insights.
            </p>
          )}
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">
            Model accuracy over the season
          </p>
          <div className="mt-3">
            <AccuracyChart data={accuracySeries(raceRows)} />
          </div>
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Championship standings</p>
          <div className="mt-3">
            <StandingsPanel
              drivers={standings.drivers}
              constructors={standings.constructors}
              teamByDriver={teamByDriver}
            />
          </div>
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Most predicted winners</p>
          <p className="mt-1 text-xs text-[var(--color-muted-text)]">
            How often each driver was the model&apos;s #1 pick this season.
          </p>
          <div className="mt-3">
            <MostPredictedWinners picks={mostPredictedWinners(picks)} />
          </div>
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Most surprising races</p>
          <div className="mt-3">
            <SurpriseCards rows={surprisingRows} />
          </div>
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Win-share distribution</p>
          <p className="mt-1 text-xs text-[var(--color-muted-text)]">
            How dominant the model&apos;s favorite looked, race by race — a right-shifted
            distribution means one-sided races.
          </p>
          <div className="mt-3">
            <WinShareHistogram bins={winShareHistogramBins(picks)} />
          </div>
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Rising &amp; fading</p>
          <div className="mt-3">
            <RisingFading trend={trend} />
          </div>
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Constructor outlook</p>
          <p className="mt-1 text-xs text-[var(--color-muted-text)]">
            Average predicted win share per race this season — higher means the model
            favors that team more often.
          </p>
          <div className="mt-3">
            <ConstructorOutlook rows={constructorOutlook(rows)} />
          </div>
        </section>
      </main>
    </>
  );
}
