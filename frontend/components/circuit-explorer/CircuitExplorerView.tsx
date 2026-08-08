"use client";

import { useEffect, useState } from "react";
import { FetchErrorMessage } from "@/components/FetchErrorMessage";
import { RankedBar } from "@/components/RankedBar";
import { StatRow } from "@/components/StatRow";
import {
  getCircuitLayout,
  getCircuitStats,
  type CircuitCatalogEntry,
  type CircuitLayout,
  type CircuitStats,
} from "@/lib/api";
import { TrackOutline } from "./TrackOutline";

const OTHER_COLOR = "#4C78A8";

export function CircuitExplorerView({ catalog }: { catalog: CircuitCatalogEntry[] }) {
  const [circuitId, setCircuitId] = useState<number>(catalog[0]?.circuit_id);
  const [layout, setLayout] = useState<CircuitLayout | null>(null);
  const [stats, setStats] = useState<CircuitStats | null>(null);
  const [winnersOpen, setWinnersOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.resolve().then(() => !cancelled && setError(null));

    function reportError(err: unknown) {
      if (cancelled) return;
      setError((prev) => prev ?? (err instanceof Error ? err.message : String(err)));
    }

    getCircuitLayout(circuitId)
      .then((r) => !cancelled && setLayout(r))
      .catch(reportError);
    getCircuitStats(circuitId)
      .then((r) => !cancelled && setStats(r))
      .catch(reportError);
    Promise.resolve().then(() => !cancelled && setWinnersOpen(false));
    return () => {
      cancelled = true;
    };
  }, [circuitId]);

  const circuit = catalog.find((c) => c.circuit_id === circuitId);
  const maxWins = Math.max(1, ...(stats?.most_wins.map((w) => w.wins) ?? [1]));

  const labelFor = (c: CircuitCatalogEntry) => `${c.name} (${c.country ?? "—"})`;

  return (
    <div className="flex flex-col gap-6">
      <select
        value={circuitId}
        onChange={(e) => setCircuitId(Number(e.target.value))}
        aria-label="Select circuit"
        className="rounded border border-[var(--color-hairline)] bg-transparent px-3 py-1.5 text-sm"
      >
        {catalog.map((c) => (
          <option key={c.circuit_id} value={c.circuit_id}>
            {labelFor(c)}
          </option>
        ))}
      </select>

      {error && <FetchErrorMessage message={`Couldn't load circuit data: ${error}`} />}

      {circuit && (
        <>
          <div>
            <h2 className="font-[family-name:var(--font-display)] text-2xl font-semibold">
              📍 {circuit.name}
            </h2>
            <p className="mt-1 text-sm text-[var(--color-muted-text)]">
              {circuit.location}, {circuit.country}
            </p>
          </div>

          {layout?.coordinates && (
            <div>
              <div className="h-72 rounded-xl bg-white/40 p-4">
                <TrackOutline coordinates={layout.coordinates} color={OTHER_COLOR} />
              </div>
              <p className="mt-1 text-xs text-[var(--color-muted-text)]">
                Track outline: {layout.attribution} (ODbL 1.0) — via{" "}
                <code>scripts/backfill_circuit_layouts.py</code>.
              </p>
            </div>
          )}

          {!stats || stats.races_held === 0 ? (
            <p className="text-sm text-[var(--color-muted-text)]">
              No race history for this circuit in the served data.
            </p>
          ) : (
            <>
              <StatRow
                stats={[
                  { label: "Races held", value: String(stats.races_held) },
                  { label: "Active", value: `${stats.first_year}–${stats.last_year}` },
                  {
                    label: "DNF rate",
                    value: stats.dnf_rate != null ? `${(stats.dnf_rate * 100).toFixed(0)}%` : "—",
                    help: "The only risk signal backed by real data in this project's source — no tire-degradation or safety-car data exists here.",
                  },
                ]}
              />

              {stats.avg_position_change != null && (
                <p className="text-sm text-[var(--color-muted-text)]">
                  🔀 <strong>Average grid → finish change: {stats.avg_position_change >= 0 ? "+" : ""}
                  {stats.avg_position_change.toFixed(1)} places</strong> — a proxy shaped by
                  strategy, reliability, and grid penalties as much as genuine overtaking
                  difficulty, not a pure &quot;how hard is this track to pass on&quot; number.
                </p>
              )}

              {stats.fastest_lap && (
                <p className="text-sm text-[var(--color-muted-text)]">
                  🏁 <strong>Fastest recorded lap: {stats.fastest_lap.time}</strong> by{" "}
                  {stats.fastest_lap.driver} ({stats.fastest_lap.year}) — reflects car
                  performance evolution since {stats.first_year}, not a fixed measure of
                  circuit difficulty; Ergast also doesn&apos;t track mid-history layout changes.
                </p>
              )}

              {stats.most_wins.length > 0 && (
                <section>
                  <p className="label-caps text-xs text-[var(--color-muted-text)]">
                    Most successful drivers here
                  </p>
                  <div className="mt-3 flex flex-col gap-2">
                    {stats.most_wins.map((w, i) => (
                      <RankedBar
                        key={w.driver}
                        label={w.driver}
                        value={w.wins}
                        maxValue={maxWins}
                        displayValue={`${w.wins}×`}
                        color={OTHER_COLOR}
                        index={i}
                      />
                    ))}
                  </div>
                </section>
              )}

              {stats.winners_by_year.length > 0 && (
                <section>
                  <button
                    type="button"
                    onClick={() => setWinnersOpen((v) => !v)}
                    className="label-caps text-left text-xs text-[var(--color-muted-text)]"
                  >
                    {winnersOpen ? "▾" : "▸"} Winners by year ({stats.winners_by_year.length} races)
                  </button>
                  {winnersOpen && (
                    <div className="mt-3 overflow-x-auto rounded-xl bg-white/40">
                      <table className="w-full min-w-max text-left text-xs">
                        <thead>
                          <tr className="border-b border-[var(--color-hairline)]">
                            {["Year", "Driver", "Constructor"].map((h) => (
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
                          {stats.winners_by_year.map((w, i) => (
                            <tr
                              // year alone isn't a unique key: some seasons (e.g.
                              // Silverstone 2020) ran two races at the same
                              // circuit, sometimes with the same winner — this
                              // list is static per circuit selection, never
                              // reordered, so the index is safe here.
                              key={`${w.year}-${i}`}
                              className="border-b border-[var(--color-hairline)] last:border-none"
                            >
                              <td className="tabular px-3 py-2">{w.year}</td>
                              <td className="px-3 py-2">{w.driver}</td>
                              <td className="px-3 py-2">{w.constructor ?? "—"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </section>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
