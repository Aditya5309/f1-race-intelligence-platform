"use client";

import { useEffect, useMemo, useState } from "react";
import { FetchErrorMessage } from "@/components/FetchErrorMessage";
import { constructorColor } from "@/lib/constructor-colors";
import {
  getDriverRaceResults,
  getDriverRadar,
  getDriverStandingsProgression,
  getDriverStats,
  getStandings,
  type DriverStats,
  type RaceResultRow,
  type StandingsProgressionRow,
  type StandingsResponse,
} from "@/lib/api";
import {
  driverLabels,
  getCareerPredictions,
  getSeasonPredictions,
  type SeasonPredictionRow,
} from "@/lib/season-aggregation";
import { StatRow } from "@/components/StatRow";
import { EntityProfileCard } from "@/components/entity-stats/EntityProfileCard";
import { TrendLineChart } from "@/components/entity-stats/TrendLineChart";
import { EntityRadarChart } from "@/components/entity-stats/EntityRadarChart";
import {
  SkeletonChart,
  SkeletonEntityProfile,
  SkeletonStatRow,
  SkeletonTable,
  SkeletonTrendChart,
} from "@/components/skeleton";

export function DriverExplorerView({ years }: { years: number[] }) {
  const [fullCareer, setFullCareer] = useState(false);
  const [year, setYear] = useState<number | null>(years[years.length - 1] ?? null);
  const [scopeRows, setScopeRows] = useState<SeasonPredictionRow[]>([]);
  const [driverId, setDriverId] = useState<number | null>(null);

  const [stats, setStats] = useState<DriverStats | null>(null);
  const [raceResults, setRaceResults] = useState<RaceResultRow[]>([]);
  const [progression, setProgression] = useState<StandingsProgressionRow[]>([]);
  const [radar, setRadar] = useState<Record<string, number> | null>(null);
  const [standings, setStandings] = useState<StandingsResponse | null>(null);
  const [scopeError, setScopeError] = useState<string | null>(null);
  const [driverError, setDriverError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const scope = fullCareer ? null : year;
  const scopeLabel = fullCareer ? `${years[0]}–${years[years.length - 1]}` : String(year);

  // Scope-level fetch: which drivers exist in this scope, plus standings
  // (independent of which driver is selected — fetched once per scope
  // change, not once per driver switch, so switching drivers never
  // re-fetches the whole season's standings table redundantly). Its own
  // error state is separate from the driver-level effect's — each effect
  // only clears the error it owns, so a real failure here can't be silently
  // wiped out just because the user switched drivers.
  useEffect(() => {
    let cancelled = false;
    Promise.resolve().then(() => !cancelled && setScopeError(null));

    function reportError(err: unknown) {
      if (cancelled) return;
      setScopeError((prev) => prev ?? (err instanceof Error ? err.message : String(err)));
    }

    const req = fullCareer ? getCareerPredictions(years) : getSeasonPredictions(year ?? years[0]);
    req
      .then((rows) => {
        if (cancelled) return;
        setScopeRows(rows);
        const labels = driverLabels(rows);
        setDriverId((current) =>
          current != null && labels.some((l) => l.driver_id === current)
            ? current
            : (labels[0]?.driver_id ?? null),
        );
      })
      .catch(reportError);
    if (scope != null) {
      getStandings(scope).then((r) => !cancelled && setStandings(r)).catch(reportError);
    } else {
      Promise.resolve().then(() => !cancelled && setStandings(null));
    }
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fullCareer, year]);

  const championshipLine = useMemo(() => {
    if (!standings || driverId == null || scope == null) return null;
    const mine = standings.drivers.find((d) => d.driver_id === driverId);
    return mine ? `Championship P${mine.position} · ${mine.points.toFixed(0)} pts (${scope})` : null;
  }, [standings, driverId, scope]);

  // Driver-specific fetch. Own error state, separate from the scope-level
  // effect above (see that effect's comment).
  useEffect(() => {
    let cancelled = false;
    if (driverId == null) {
      Promise.resolve().then(() => !cancelled && setIsLoading(false));
      return () => {
        cancelled = true;
      };
    }
    Promise.resolve().then(() => !cancelled && setIsLoading(true));
    Promise.resolve().then(() => !cancelled && setDriverError(null));
    // Clear the previous driver's fetch-derived state so a failed fetch
    // can't leave stale data rendered under the new driver's heading (the
    // heading itself comes from `labels`/`driverRows`, which are always in
    // sync with `driverId` via `scopeRows`).
    Promise.resolve().then(() => !cancelled && setStats(null));
    Promise.resolve().then(() => !cancelled && setRaceResults([]));
    Promise.resolve().then(() => !cancelled && setProgression([]));
    Promise.resolve().then(() => !cancelled && setRadar(null));

    function reportError(err: unknown) {
      if (cancelled) return;
      setDriverError((prev) => prev ?? (err instanceof Error ? err.message : String(err)));
    }

    const statsReq = getDriverStats(driverId, scope ?? undefined)
      .then((r) => !cancelled && setStats(r))
      .catch(reportError);
    const raceResultsReq = getDriverRaceResults(driverId, scope ?? undefined)
      .then((r) => !cancelled && setRaceResults(r))
      .catch(reportError);
    let progressionReq: Promise<unknown>;
    let radarReq: Promise<unknown>;
    if (scope != null) {
      progressionReq = getDriverStandingsProgression(driverId, scope)
        .then((r) => !cancelled && setProgression(r))
        .catch(reportError);
      radarReq = getDriverRadar(driverId, scope).then((r) => !cancelled && setRadar(r)).catch(reportError);
    } else {
      // Career scope: reset season-only state asynchronously (not a
      // synchronous setState-in-effect) — these fields are also
      // render-gated on scope, so this is a cleanup, not a display fix.
      progressionReq = radarReq = Promise.resolve().then(() => {
        if (cancelled) return;
        setProgression([]);
        setRadar(null);
      });
    }
    Promise.allSettled([statsReq, raceResultsReq, progressionReq, radarReq]).then(
      () => !cancelled && setIsLoading(false),
    );
    return () => {
      cancelled = true;
    };
  }, [driverId, scope]);

  const labels = useMemo(() => driverLabels(scopeRows), [scopeRows]);
  const driverRows = useMemo(
    () => scopeRows.filter((r) => r.driver_id === driverId).sort((a, b) => a.round - b.round),
    [scopeRows, driverId],
  );
  const team = driverRows.find((r) => r.constructor_name)?.constructor_name ?? null;
  const teamColor = constructorColor(team);
  const label = labels.find((l) => l.driver_id === driverId)?.label ?? "";

  const winsInScope = driverRows.filter((r) => r.driver_id === r.actual_winner_driver_id).length;
  const avgPredictedRank = driverRows.length
    ? driverRows.reduce((sum, r) => sum + r.predicted_rank, 0) / driverRows.length
    : null;
  const bestRank = driverRows.length ? Math.min(...driverRows.map((r) => r.predicted_rank)) : null;

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={fullCareer}
            onChange={(e) => setFullCareer(e.target.checked)}
          />
          Full career (all seasons)
        </label>
        {!fullCareer && (
          <select
            value={year ?? years[years.length - 1]}
            onChange={(e) => setYear(Number(e.target.value))}
            aria-label="Select season"
            className="rounded border border-[var(--color-hairline)] bg-transparent px-3 py-1.5 text-sm"
          >
            {[...years].reverse().map((y) => (
              <option key={y} value={y}>
                {y}
              </option>
            ))}
          </select>
        )}
        <select
          value={driverId ?? ""}
          onChange={(e) => setDriverId(Number(e.target.value))}
          aria-label="Select driver"
          className="rounded border border-[var(--color-hairline)] bg-transparent px-3 py-1.5 text-sm"
        >
          {labels.map((l) => (
            <option key={l.driver_id} value={l.driver_id}>
              {l.label}
            </option>
          ))}
        </select>
      </div>

      {scopeError && <FetchErrorMessage message={`Couldn't load season data: ${scopeError}`} />}
      {driverError && <FetchErrorMessage message={`Couldn't load driver data: ${driverError}`} />}

      {driverId != null &&
        (isLoading ? (
          <>
            <SkeletonEntityProfile label="Loading driver profile…" />
            <SkeletonStatRow columns={6} label="Loading career stats…" />
            <SkeletonStatRow columns={4} label="Loading scope stats…" />

            <section>
              <p className="label-caps text-xs text-[var(--color-muted-text)]">Trends</p>
              <div className="mt-3 grid gap-6 sm:grid-cols-2">
                <SkeletonTrendChart label="Loading qualifying trend…" />
                <SkeletonTrendChart label="Loading finishing trend…" />
              </div>
              <div className="mt-6 grid gap-6 sm:grid-cols-2">
                <SkeletonTrendChart label="Loading points progression…" />
                <SkeletonTrendChart label="Loading win probability trend…" />
              </div>
            </section>

            <section>
              <p className="label-caps text-xs text-[var(--color-muted-text)]">Season profile</p>
              <div className="mt-3">
                <SkeletonChart height="h-72" variant="radar" label="Loading season profile…" />
              </div>
            </section>

            <section>
              <p className="label-caps text-xs text-[var(--color-muted-text)]">Race log</p>
              <div className="mt-3">
                <SkeletonTable rows={6} columns={5} label="Loading race log…" />
              </div>
            </section>
          </>
        ) : (
          <>
            <EntityProfileCard
              color={teamColor}
              name={label}
              subtitle={`${team ?? "Unknown team"} · ${scopeLabel} · ${driverRows.length} race(s)`}
              championshipLine={championshipLine}
            />

            {stats && (
              <StatRow
                columns={6}
                stats={[
                  { label: "Wins", value: String(stats.wins) },
                  { label: "Podiums", value: String(stats.podiums) },
                  { label: "Poles", value: stats.poles != null ? String(stats.poles) : "—" },
                  { label: "Points", value: stats.points.toFixed(0) },
                  {
                    label: "Avg qualifying",
                    value: stats.avg_quali != null ? `P${stats.avg_quali.toFixed(1)}` : "—",
                  },
                  {
                    label: "Avg finish",
                    value: stats.avg_finish != null ? `P${stats.avg_finish.toFixed(1)}` : "—",
                  },
                ]}
              />
            )}

            <StatRow
              columns={4}
              stats={[
                { label: "Races (scope)", value: String(driverRows.length) },
                {
                  label: "Avg predicted rank",
                  value: avgPredictedRank != null ? avgPredictedRank.toFixed(1) : "—",
                },
                { label: "Best rank", value: bestRank != null ? String(bestRank) : "—" },
                { label: "Actual wins (scope)", value: String(winsInScope) },
              ]}
            />

            <section>
              <p className="label-caps text-xs text-[var(--color-muted-text)]">Trends</p>
              <div className="mt-3 grid gap-6 sm:grid-cols-2">
                <TrendLineChart
                  data={raceResults
                    .filter((r) => r.quali_position != null)
                    .map((r) => ({
                      race: `${r.year} R${r.round}`,
                      quali_position: r.quali_position,
                    }))}
                  xKey="race"
                  yKey="quali_position"
                  color={teamColor}
                  invertY
                  emptyMessage="No qualifying history in this scope."
                />
                <TrendLineChart
                  data={raceResults
                    .filter((r) => r.finish != null)
                    .map((r) => ({ race: `${r.year} R${r.round}`, finish: r.finish }))}
                  xKey="race"
                  yKey="finish"
                  color={teamColor}
                  invertY
                  emptyMessage="No finishing history in this scope."
                />
              </div>
              <div className="mt-6 grid gap-6 sm:grid-cols-2">
                {scope != null ? (
                  <TrendLineChart
                    data={progression.map((p) => ({ round: `R${p.round}`, points: p.points }))}
                    xKey="round"
                    yKey="points"
                    color={teamColor}
                    emptyMessage="Standings progression unavailable."
                  />
                ) : (
                  <p className="text-sm text-[var(--color-muted-text)]">
                    Championship points chart is season-scoped — turn off &quot;Full
                    career&quot; to see it.
                  </p>
                )}
                <TrendLineChart
                  data={driverRows.map((r) => ({
                    race: `${r.year} R${r.round}`,
                    win_probability: r.win_probability,
                  }))}
                  xKey="race"
                  yKey="win_probability"
                  color={teamColor}
                  emptyMessage="No prediction history in this scope."
                />
              </div>
            </section>

            <section>
              <p className="label-caps text-xs text-[var(--color-muted-text)]">Season profile</p>
              {scope == null ? (
                <p className="mt-2 text-sm text-[var(--color-muted-text)]">
                  Season profile is season-scoped only — turn off &quot;Full career&quot; to see
                  it.
                </p>
              ) : radar ? (
                <>
                  <div className="mt-3">
                    <EntityRadarChart series={[{ label, color: teamColor, scores: radar }]} />
                  </div>
                  <p className="mt-2 text-xs text-[var(--color-muted-text)]">
                    Scores are relative to this season&apos;s field (0 = worst, 100 = best among
                    drivers with 3+ races) — not an absolute or cross-era scale.
                  </p>
                </>
              ) : (
                <p className="mt-2 text-sm text-[var(--color-muted-text)]">
                  Not enough races this season for a season profile.
                </p>
              )}
            </section>

            <section>
              <p className="label-caps text-xs text-[var(--color-muted-text)]">Race log</p>
              <div className="mt-3 overflow-x-auto rounded-xl bg-white/40">
                <table className="w-full min-w-max text-left text-xs">
                  <thead>
                    <tr className="border-b border-[var(--color-hairline)]">
                      {["Year", "Round", "Predicted rank", "Win share", "Result"].map((h) => (
                        <th
                          key={h}
                          className="label-caps px-3 py-2 font-normal text-[var(--color-muted-text)]"
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {driverRows.map((r) => (
                      <tr
                        key={r.race_id}
                        className="border-b border-[var(--color-hairline)] last:border-none"
                      >
                        <td className="tabular px-3 py-2">{r.year}</td>
                        <td className="tabular px-3 py-2">{r.round}</td>
                        <td className="tabular px-3 py-2">{r.predicted_rank}</td>
                        <td className="tabular px-3 py-2">
                          {(r.win_probability * 100).toFixed(1)}%
                        </td>
                        <td className="px-3 py-2">
                          {r.driver_id === r.actual_winner_driver_id
                            ? "🏆 Won"
                            : r.actual_winner_driver_id == null
                              ? "—"
                              : ""}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        ))}
    </div>
  );
}
