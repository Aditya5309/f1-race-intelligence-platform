"use client";

import type { RaceSummary, UpcomingRace } from "@/lib/api";

interface PickerEntry {
  race_id: number;
  year: number;
  round: number;
  isUpcoming: boolean;
}

export function RacePicker({
  races,
  upcoming,
  selectedRaceId,
  onSelect,
}: {
  races: RaceSummary[];
  upcoming: UpcomingRace | null;
  selectedRaceId: number;
  onSelect: (raceId: number) => void;
}) {
  const entries: PickerEntry[] = [
    ...races.map((r) => ({ ...r, isUpcoming: false })),
    ...(upcoming
      ? [{ race_id: upcoming.race_id, year: upcoming.year, round: upcoming.round, isUpcoming: true }]
      : []),
  ].sort((a, b) => a.year - b.year || a.round - b.round);

  const current = entries.find((e) => e.race_id === selectedRaceId) ?? entries[entries.length - 1];
  const years = Array.from(new Set(entries.map((e) => e.year))).sort((a, b) => b - a);
  const yearEntries = entries.filter((e) => e.year === current?.year);
  const pos = entries.findIndex((e) => e.race_id === selectedRaceId);

  return (
    <div className="flex flex-wrap items-center gap-3">
      <button
        type="button"
        disabled={pos <= 0}
        onClick={() => onSelect(entries[pos - 1].race_id)}
        className="label-caps rounded border border-[var(--color-hairline)] px-2 py-1 text-xs disabled:opacity-30"
      >
        ◀ Prev
      </button>
      <button
        type="button"
        disabled={pos === -1 || pos >= entries.length - 1}
        onClick={() => onSelect(entries[pos + 1].race_id)}
        className="label-caps rounded border border-[var(--color-hairline)] px-2 py-1 text-xs disabled:opacity-30"
      >
        Next ▶
      </button>

      <select
        value={current?.year}
        onChange={(e) => {
          const year = Number(e.target.value);
          const first = entries.find((entry) => entry.year === year);
          if (first) onSelect(first.race_id);
        }}
        aria-label="Select season"
        className="rounded border border-[var(--color-hairline)] bg-transparent px-2 py-1 text-sm"
      >
        {years.map((y) => (
          <option key={y} value={y}>
            {y}
          </option>
        ))}
      </select>

      <select
        value={selectedRaceId}
        onChange={(e) => onSelect(Number(e.target.value))}
        aria-label="Select round"
        className="rounded border border-[var(--color-hairline)] bg-transparent px-2 py-1 text-sm"
      >
        {yearEntries.map((e) => (
          <option key={e.race_id} value={e.race_id}>
            {e.isUpcoming ? `🔮 Round ${e.round} (upcoming)` : `Round ${e.round}`}
          </option>
        ))}
      </select>
    </div>
  );
}
