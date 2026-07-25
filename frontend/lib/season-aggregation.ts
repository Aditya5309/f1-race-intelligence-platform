/**
 * Client-side aggregation over GET /races + GET /predictions/{race_id} — the
 * TypeScript port of app/views/common.py::season_predictions() and the
 * derived computations in app/views/season_analytics.py. No dedicated
 * aggregation endpoint exists (by design, see the roadmap's Season
 * Analytics data-dependency audit) — every function here operates on the
 * same flattened one-row-per-driver-per-race shape the Streamlit page
 * computes from.
 */

import { getPredictions, listRaces, type DriverPrediction } from "./api";

export interface SeasonPredictionRow extends DriverPrediction {
  race_id: number;
  year: number;
  round: number;
  actual_winner_driver_id: number | null;
  model_top1_hit: boolean | null;
}

export async function getSeasonPredictions(year: number): Promise<SeasonPredictionRow[]> {
  const races = await listRaces(year);
  const perRace = await Promise.all(
    races.map((race) =>
      getPredictions(race.race_id).then((body) =>
        body.predictions.map((pred) => ({
          ...pred,
          race_id: body.race_id,
          year: body.year,
          round: body.round,
          actual_winner_driver_id: body.actual_winner_driver_id,
          model_top1_hit: body.model_top1_hit,
        })),
      ),
    ),
  );
  return perRace.flat();
}

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/** One row per race_id (first occurrence) — mirrors frame.drop_duplicates("race_id"). */
export function byRace(rows: SeasonPredictionRow[]): SeasonPredictionRow[] {
  const seen = new Set<number>();
  const out: SeasonPredictionRow[] = [];
  for (const row of rows) {
    if (!seen.has(row.race_id)) {
      seen.add(row.race_id);
      out.push(row);
    }
  }
  return out;
}

export function hitRate(byRaceRows: SeasonPredictionRow[]): number | null {
  const decided = byRaceRows.filter((r) => r.model_top1_hit !== null);
  if (!decided.length) return null;
  return decided.filter((r) => r.model_top1_hit).length / decided.length;
}

export function avgFieldSize(rows: SeasonPredictionRow[]): number {
  const counts = new Map<number, number>();
  for (const r of rows) counts.set(r.race_id, (counts.get(r.race_id) ?? 0) + 1);
  const values = [...counts.values()];
  return values.length ? mean(values) : 0;
}

export function topPicks(rows: SeasonPredictionRow[]): SeasonPredictionRow[] {
  return rows.filter((r) => r.predicted_rank === 1);
}

/** Races the model's favorite lost, worst misses first (highest predicted_rank for the actual winner). */
export function surprises(rows: SeasonPredictionRow[], n = 3): SeasonPredictionRow[] {
  const winners = rows.filter((r) => r.driver_id === r.actual_winner_driver_id);
  return [...winners].sort((a, b) => b.predicted_rank - a.predicted_rank).slice(0, n);
}

export interface DriverTrendRow {
  driver_id: number;
  label: string;
  early_avg: number;
  recent_avg: number;
  delta: number; // negative = improving (predicted rank getting lower)
}

const MIN_RACES_FOR_TREND = 4;

export function driverTrend(rows: SeasonPredictionRow[]): DriverTrendRow[] {
  const byDriver = new Map<number, SeasonPredictionRow[]>();
  for (const r of rows) {
    const list = byDriver.get(r.driver_id) ?? [];
    list.push(r);
    byDriver.set(r.driver_id, list);
  }
  const out: DriverTrendRow[] = [];
  for (const [driverId, group] of byDriver) {
    if (group.length < MIN_RACES_FOR_TREND) continue;
    const sorted = [...group].sort((a, b) => a.round - b.round);
    const mid = Math.floor(sorted.length / 2);
    const earlyAvg = mean(sorted.slice(0, mid).map((r) => r.predicted_rank));
    const recentAvg = mean(sorted.slice(mid).map((r) => r.predicted_rank));
    const namedRow = [...group].reverse().find((r) => r.driver_name);
    out.push({
      driver_id: driverId,
      label: namedRow?.driver_name ?? String(driverId),
      early_avg: earlyAvg,
      recent_avg: recentAvg,
      delta: recentAvg - earlyAvg,
    });
  }
  return out;
}

export interface PickCount {
  label: string;
  team: string | null;
  picks: number;
}

export function mostPredictedWinners(picks: SeasonPredictionRow[]): PickCount[] {
  const byLabel = new Map<string, PickCount>();
  for (const r of picks) {
    const label = r.driver_name ?? String(r.driver_id);
    const entry = byLabel.get(label) ?? { label, team: null, picks: 0 };
    entry.picks += 1;
    entry.team = r.constructor_name ?? entry.team;
    byLabel.set(label, entry);
  }
  return [...byLabel.values()].sort((a, b) => b.picks - a.picks);
}

export interface AccuracyPoint {
  round: number;
  hit: number;
  cumulative: number;
}

export function accuracySeries(byRaceRows: SeasonPredictionRow[]): AccuracyPoint[] {
  const decided = byRaceRows
    .filter((r) => r.model_top1_hit !== null)
    .sort((a, b) => a.round - b.round);
  let hits = 0;
  return decided.map((r, i) => {
    if (r.model_top1_hit) hits += 1;
    return { round: r.round, hit: r.model_top1_hit ? 1 : 0, cumulative: hits / (i + 1) };
  });
}

export interface ConstructorOutlookRow {
  constructor_name: string;
  win_probability: number;
}

export function constructorOutlook(rows: SeasonPredictionRow[], topN = 8): ConstructorOutlookRow[] {
  const byConstructor = new Map<string, number[]>();
  for (const r of rows) {
    if (!r.constructor_name) continue;
    const list = byConstructor.get(r.constructor_name) ?? [];
    list.push(r.win_probability);
    byConstructor.set(r.constructor_name, list);
  }
  return [...byConstructor.entries()]
    .map(([constructor_name, values]) => ({ constructor_name, win_probability: mean(values) }))
    .sort((a, b) => b.win_probability - a.win_probability)
    .slice(0, topN);
}

export interface HistogramBin {
  bin: string;
  count: number;
}

export function winShareHistogramBins(picks: SeasonPredictionRow[], bins = 10): HistogramBin[] {
  const counts = new Array(bins).fill(0);
  for (const r of picks) {
    const idx = Math.min(bins - 1, Math.max(0, Math.floor(r.win_probability * bins)));
    counts[idx] += 1;
  }
  return counts.map((count, i) => ({
    bin: `${Math.round((i / bins) * 100)}–${Math.round(((i + 1) / bins) * 100)}%`,
    count,
  }));
}

const MIN_DOMINANCE_PICKS = 2;

/** 3-4 deterministic, templated one-line insights re-narrating numbers this
 * page already computes elsewhere — no LLM, no new data (matches
 * app/views/season_analytics.py::_season_insights exactly). */
export function seasonInsights(
  picks: SeasonPredictionRow[],
  surprisingRows: SeasonPredictionRow[],
  trend: DriverTrendRow[],
): string[] {
  const insights: string[] = [];

  const byLabel = new Map<string, number[]>();
  for (const r of picks) {
    const label = r.driver_name ?? String(r.driver_id);
    const list = byLabel.get(label) ?? [];
    list.push(r.win_probability);
    byLabel.set(label, list);
  }
  const dominance = [...byLabel.entries()]
    .map(([label, values]) => ({ label, mean: mean(values), size: values.length }))
    .filter((d) => d.size >= MIN_DOMINANCE_PICKS)
    .sort((a, b) => b.mean - a.mean);
  if (dominance.length) {
    const d = dominance[0];
    insights.push(
      `Most dominant: ${d.label} averaged ${(d.mean * 100).toFixed(1)}% win share across ${d.size} races as the model's favorite.`,
    );
  }

  if (surprisingRows.length) {
    const row = surprisingRows[0];
    insights.push(
      `Biggest surprise: ${row.driver_name ?? `driver ${row.driver_id}`} won from model rank #${row.predicted_rank}.`,
    );
  }

  if (trend.length) {
    const best = [...trend].sort((a, b) => a.delta - b.delta)[0];
    insights.push(
      `Most improved: ${best.label}, average predicted rank ${best.early_avg.toFixed(1)} → ${best.recent_avg.toFixed(1)} this season.`,
    );
  }

  if (picks.length) {
    const byConfidence = [...picks].sort((a, b) => b.win_probability - a.win_probability);
    const mostConf = byConfidence[0];
    const leastConf = byConfidence[byConfidence.length - 1];
    insights.push(
      `Most confident pick: ${mostConf.driver_name ?? mostConf.driver_id} at ` +
        `${(mostConf.win_probability * 100).toFixed(1)}% (round ${mostConf.round}). ` +
        `Least confident: ${leastConf.driver_name ?? leastConf.driver_id} at ` +
        `${(leastConf.win_probability * 100).toFixed(1)}% (round ${leastConf.round}).`,
    );
  }

  return insights;
}
