"use client";

import { useEffect, useMemo, useState } from "react";
import { constructorColor } from "@/lib/constructor-colors";
import {
  getConstructorDrivers,
  getConstructorRaceResults,
  getConstructorStats,
  getConstructors,
  getStandings,
  type ConstructorCatalogEntry,
  type ConstructorStats,
  type RaceResultRow,
} from "@/lib/api";
import { getCareerPredictions, getSeasonPredictions, type SeasonPredictionRow } from "@/lib/season-aggregation";
import { StatRow } from "@/components/StatRow";
import { EntityProfileCard } from "@/components/entity-stats/EntityProfileCard";
import { TrendLineChart } from "@/components/entity-stats/TrendLineChart";

export function TeamView({ years }: { years: number[] }) {
  const [catalog, setCatalog] = useState<ConstructorCatalogEntry[]>([]);
  const [fullCareer, setFullCareer] = useState(false);
  const [year, setYear] = useState<number | null>(years[years.length - 1] ?? null);
  const [constructorId, setConstructorId] = useState<number | null>(null);
  const [frame, setFrame] = useState<SeasonPredictionRow[]>([]);

  const [stats, setStats] = useState<ConstructorStats | null>(null);
  const [raceResults, setRaceResults] = useState<RaceResultRow[]>([]);
  const [drivers, setDrivers] = useState<string[]>([]);
  const [championshipLine, setChampionshipLine] = useState<string | null>(null);

  const scope = fullCareer ? null : year;
  const scopeLabel = fullCareer ? `${years[0]}–${years[years.length - 1]}` : String(year);

  useEffect(() => {
    getConstructors().then((c) => {
      setCatalog(c);
      setConstructorId((current) => current ?? c[0]?.constructor_id ?? null);
    });
  }, []);

  useEffect(() => {
    let cancelled = false;
    const req = fullCareer ? getCareerPredictions(years) : getSeasonPredictions(year ?? years[0]);
    req.then((rows) => !cancelled && setFrame(rows));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fullCareer, year]);

  useEffect(() => {
    if (constructorId == null) return;
    let cancelled = false;
    getConstructorStats(constructorId, scope ?? undefined).then((r) => !cancelled && setStats(r));
    getConstructorRaceResults(constructorId, scope ?? undefined).then(
      (r) => !cancelled && setRaceResults(r),
    );
    if (scope != null) {
      getConstructorDrivers(constructorId, scope).then((r) => !cancelled && setDrivers(r));
      getStandings(scope).then((r) => {
        if (cancelled) return;
        const mine = r.constructors.find((c) => c.constructor_id === constructorId);
        setChampionshipLine(
          mine ? `Championship P${mine.position} · ${mine.points.toFixed(0)} pts (${scope})` : null,
        );
      });
    } else {
      Promise.resolve().then(() => {
        if (cancelled) return;
        setDrivers([]);
        setChampionshipLine(null);
      });
    }
    return () => {
      cancelled = true;
    };
  }, [constructorId, scope]);

  const teamName = catalog.find((c) => c.constructor_id === constructorId)?.name ?? "";
  const nationality = catalog.find((c) => c.constructor_id === constructorId)?.nationality ?? "Unknown";
  const teamColor = constructorColor(teamName);

  const teamFrame = useMemo(
    () => frame.filter((r) => r.constructor_name === teamName),
    [frame, teamName],
  );
  const picks = teamFrame.filter((r) => r.predicted_rank === 1);
  const decided = picks.filter((r) => r.actual_winner_driver_id != null).length;
  const hits = picks.filter((r) => r.driver_id === r.actual_winner_driver_id).length;

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
          value={constructorId ?? ""}
          onChange={(e) => setConstructorId(Number(e.target.value))}
          className="rounded border border-[var(--color-hairline)] bg-transparent px-3 py-1.5 text-sm"
        >
          {catalog.map((c) => (
            <option key={c.constructor_id} value={c.constructor_id}>
              {c.name}
            </option>
          ))}
        </select>
      </div>

      {constructorId != null && (
        <>
          <EntityProfileCard
            color={teamColor}
            name={teamName}
            subtitle={`${nationality} · ${scopeLabel}${drivers.length ? ` · ${drivers.join(", ")}` : ""}`}
            championshipLine={championshipLine}
          />

          {stats && stats.races > 0 ? (
            <StatRow
              columns={5}
              stats={[
                { label: "Races", value: String(stats.races) },
                { label: "Wins", value: String(stats.wins) },
                { label: "Podiums", value: String(stats.podiums) },
                { label: "Points", value: stats.points.toFixed(0) },
                {
                  label: "Avg qualifying",
                  value: stats.avg_quali != null ? `P${stats.avg_quali.toFixed(1)}` : "—",
                },
              ]}
            />
          ) : (
            <p className="text-sm text-[var(--color-muted-text)]">
              No historical stats for this team in this scope.
            </p>
          )}

          {teamFrame.length > 0 && (
            <p className="text-sm text-[var(--color-muted-text)]">
              🎯 The model picked a {teamName} driver as its #1 favorite in {picks.length} race(s)
              this scope
              {decided > 0
                ? `; that pick won ${hits} of ${decided} decided race(s) (${((hits / decided) * 100).toFixed(0)}%).`
                : picks.length > 0
                  ? "; no results decided yet."
                  : "."}
            </p>
          )}

          <section>
            <p className="label-caps text-xs text-[var(--color-muted-text)]">Trends</p>
            <div className="mt-3 grid gap-6 sm:grid-cols-2">
              <TrendLineChart
                data={raceResults
                  .filter((r) => r.quali_position != null)
                  .map((r) => ({ race: `${r.year} R${r.round}`, quali_position: r.quali_position }))}
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
          </section>
        </>
      )}
    </div>
  );
}
