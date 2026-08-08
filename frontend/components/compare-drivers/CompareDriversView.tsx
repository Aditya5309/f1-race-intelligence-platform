"use client";

import { useEffect, useMemo, useState } from "react";
import { FetchErrorMessage } from "@/components/FetchErrorMessage";
import { constructorColor } from "@/lib/constructor-colors";
import { getDriverRadar, getDriverStats, type DriverStats } from "@/lib/api";
import { driverLabels, getSeasonPredictions, type SeasonPredictionRow } from "@/lib/season-aggregation";
import { EntityRadarChart, type RadarSeries } from "@/components/entity-stats/EntityRadarChart";

function DriverStatColumn({
  label,
  team,
  stats,
}: {
  label: string;
  team: string | null;
  stats: DriverStats | null;
}) {
  const color = constructorColor(team);
  return (
    <div className="rounded-2xl border border-[var(--color-hairline)] bg-white/40 p-6">
      <div className="flex items-center gap-2">
        <span aria-hidden className="h-3 w-3 rounded-full" style={{ backgroundColor: color }} />
        <h3 className="text-lg font-semibold">{label}</h3>
      </div>
      <p className="text-xs text-[var(--color-muted-text)]">{team ?? "Unknown team"}</p>
      {!stats || stats.races === 0 ? (
        <p className="mt-4 text-sm text-[var(--color-muted-text)]">
          No stats for this driver this season.
        </p>
      ) : (
        <div className="mt-4 grid grid-cols-2 gap-4">
          <div>
            <p className="tabular text-xl font-semibold">
              {stats.avg_quali != null ? `P${stats.avg_quali.toFixed(1)}` : "—"}
            </p>
            <p className="label-caps text-[10px] text-[var(--color-muted-text)]">Avg qualifying</p>
          </div>
          <div>
            <p className="tabular text-xl font-semibold">
              {stats.avg_finish != null ? `P${stats.avg_finish.toFixed(1)}` : "—"}
            </p>
            <p className="label-caps text-[10px] text-[var(--color-muted-text)]">Avg finish</p>
          </div>
          {stats.consistency_std != null && (
            <div className="col-span-2">
              <p className="tabular text-xl font-semibold">±{stats.consistency_std.toFixed(1)}</p>
              <p className="label-caps text-[10px] text-[var(--color-muted-text)]">
                Consistency (std dev of finish position)
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function CompareDriversView({ years }: { years: number[] }) {
  const [year, setYear] = useState(years[0]);
  const [rows, setRows] = useState<SeasonPredictionRow[]>([]);
  const [driverA, setDriverA] = useState<number | null>(null);
  const [driverB, setDriverB] = useState<number | null>(null);
  const [raceId, setRaceId] = useState<number | null>(null);
  const [statsA, setStatsA] = useState<DriverStats | null>(null);
  const [statsB, setStatsB] = useState<DriverStats | null>(null);
  const [radarA, setRadarA] = useState<Record<string, number> | null>(null);
  const [radarB, setRadarB] = useState<Record<string, number> | null>(null);
  const [seasonError, setSeasonError] = useState<string | null>(null);
  const [driversError, setDriversError] = useState<string | null>(null);

  // Each of this component's 2 effects owns its own error state — only
  // clears the error it set itself, so a real failure in one effect can't
  // be silently wiped out just because the other effect re-ran.
  useEffect(() => {
    let cancelled = false;
    Promise.resolve().then(() => !cancelled && setSeasonError(null));
    getSeasonPredictions(year)
      .then((r) => {
        if (cancelled) return;
        setRows(r);
        const labels = driverLabels(r);
        setDriverA(labels[0]?.driver_id ?? null);
        setDriverB(labels[1]?.driver_id ?? null);
        const races = [...new Set(r.map((row) => row.race_id))];
        setRaceId(races[0] ?? null);
      })
      .catch((err) => {
        if (!cancelled) setSeasonError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [year]);

  useEffect(() => {
    let cancelled = false;
    Promise.resolve().then(() => !cancelled && setDriversError(null));

    function reportError(err: unknown) {
      if (cancelled) return;
      setDriversError((prev) => prev ?? (err instanceof Error ? err.message : String(err)));
    }

    if (driverA != null) {
      getDriverStats(driverA, year).then((r) => !cancelled && setStatsA(r)).catch(reportError);
      getDriverRadar(driverA, year).then((r) => !cancelled && setRadarA(r)).catch(reportError);
    }
    if (driverB != null) {
      getDriverStats(driverB, year).then((r) => !cancelled && setStatsB(r)).catch(reportError);
      getDriverRadar(driverB, year).then((r) => !cancelled && setRadarB(r)).catch(reportError);
    }
    return () => {
      cancelled = true;
    };
  }, [driverA, driverB, year]);

  const labels = useMemo(() => driverLabels(rows), [rows]);
  const races = useMemo(
    () =>
      [...new Map(rows.map((r) => [r.race_id, r.round])).entries()]
        .sort((a, b) => a[1] - b[1])
        .map(([race_id, round]) => ({ race_id, round })),
    [rows],
  );

  const teamOf = (driverId: number | null) =>
    rows.find((r) => r.driver_id === driverId)?.constructor_name ?? null;
  const labelOf = (driverId: number | null) =>
    labels.find((l) => l.driver_id === driverId)?.label ?? "";

  const raceRowFor = (driverId: number | null) =>
    rows.find((r) => r.race_id === raceId && r.driver_id === driverId) ?? null;

  const radarSeries: RadarSeries[] = [];
  if (radarA && driverA != null) {
    radarSeries.push({ label: labelOf(driverA), color: constructorColor(teamOf(driverA)), scores: radarA });
  }
  if (radarB && driverB != null) {
    radarSeries.push({ label: labelOf(driverB), color: constructorColor(teamOf(driverB)), scores: radarB });
  }

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-wrap items-center gap-3">
        <select
          value={year}
          onChange={(e) => setYear(Number(e.target.value))}
          aria-label="Select season"
          className="rounded border border-[var(--color-hairline)] bg-transparent px-3 py-1.5 text-sm"
        >
          {years.map((y) => (
            <option key={y} value={y}>
              {y}
            </option>
          ))}
        </select>
      </div>

      {seasonError && <FetchErrorMessage message={`Couldn't load season data: ${seasonError}`} />}
      {driversError && (
        <FetchErrorMessage message={`Couldn't load comparison data: ${driversError}`} />
      )}

      {labels.length < 2 ? (
        <p className="text-sm text-[var(--color-muted-text)]">
          Not enough drivers this season to compare.
        </p>
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-2">
            <select
              value={driverA ?? ""}
              onChange={(e) => setDriverA(Number(e.target.value))}
              aria-label="Select first driver"
              className="rounded border border-[var(--color-hairline)] bg-transparent px-3 py-1.5 text-sm"
            >
              {labels.map((l) => (
                <option key={l.driver_id} value={l.driver_id}>
                  {l.label}
                </option>
              ))}
            </select>
            <select
              value={driverB ?? ""}
              onChange={(e) => setDriverB(Number(e.target.value))}
              aria-label="Select second driver"
              className="rounded border border-[var(--color-hairline)] bg-transparent px-3 py-1.5 text-sm"
            >
              {labels.map((l) => (
                <option key={l.driver_id} value={l.driver_id}>
                  {l.label}
                </option>
              ))}
            </select>
          </div>

          {driverA === driverB ? (
            <p className="text-sm text-[var(--color-accent-red)]">
              Pick two different drivers to compare.
            </p>
          ) : (
            <>
              <div className="grid gap-6 sm:grid-cols-2">
                <DriverStatColumn label={labelOf(driverA)} team={teamOf(driverA)} stats={statsA} />
                <DriverStatColumn label={labelOf(driverB)} team={teamOf(driverB)} stats={statsB} />
              </div>

              <section>
                <p className="label-caps text-xs text-[var(--color-muted-text)]">
                  Predicted win share for a race this season
                </p>
                <select
                  value={raceId ?? ""}
                  onChange={(e) => setRaceId(Number(e.target.value))}
                  aria-label="Select race"
                  className="mt-3 rounded border border-[var(--color-hairline)] bg-transparent px-3 py-1.5 text-sm"
                >
                  {races.map((r) => (
                    <option key={r.race_id} value={r.race_id}>
                      Round {r.round}
                    </option>
                  ))}
                </select>
                <div className="mt-3 grid gap-4 sm:grid-cols-2">
                  {[driverA, driverB].map((driverId) => {
                    const row = raceRowFor(driverId);
                    return (
                      <div key={driverId} className="rounded-xl bg-white/40 p-4">
                        <p className="text-sm font-medium">{labelOf(driverId)}</p>
                        {row ? (
                          <p className="tabular mt-1 text-2xl font-semibold">
                            {(row.win_probability * 100).toFixed(1)}%
                          </p>
                        ) : (
                          <p className="mt-1 text-xs text-[var(--color-muted-text)]">
                            Didn&apos;t enter this race.
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </section>

              <section>
                <p className="label-caps text-xs text-[var(--color-muted-text)]">Season profile</p>
                {radarSeries.length ? (
                  <>
                    <div className="mt-3">
                      <EntityRadarChart series={radarSeries} />
                    </div>
                    <p className="mt-2 text-xs text-[var(--color-muted-text)]">
                      Scores are relative to this season&apos;s field (0 = worst, 100 = best among
                      drivers with 3+ races) — not an absolute or cross-era scale.
                    </p>
                  </>
                ) : (
                  <p className="mt-2 text-sm text-[var(--color-muted-text)]">
                    Not enough races this season for a season profile for either driver.
                  </p>
                )}
              </section>
            </>
          )}
        </>
      )}
    </div>
  );
}
