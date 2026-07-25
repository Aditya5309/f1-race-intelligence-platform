import { constructorColor } from "@/lib/constructor-colors";
import type { ConstructorStanding, DriverStanding } from "@/lib/api";
import { RankedBar } from "./RankedBar";

/**
 * GET /standings/{year} (Decision 062) — the one new backend route this
 * phase needed. Driver rows carry no constructor id of their own (same
 * Ergast-schema separation app/views/metadata.py::season_driver_standings
 * lives with) — teamByDriver, derived from this season's predictions, fills
 * in the color the same way the Streamlit page does.
 */
export function StandingsPanel({
  drivers,
  constructors,
  teamByDriver,
}: {
  drivers: DriverStanding[];
  constructors: ConstructorStanding[];
  teamByDriver: Map<number, string | null>;
}) {
  const maxDriverPoints = Math.max(1, ...drivers.map((d) => d.points));
  const maxConstructorPoints = Math.max(1, ...constructors.map((c) => c.points));

  return (
    <div className="grid gap-8 sm:grid-cols-2">
      <div className="flex flex-col gap-2">
        <p className="label-caps text-xs text-[var(--color-muted-text)]">Drivers</p>
        {drivers.length ? (
          drivers.slice(0, 10).map((d, i) => (
            <RankedBar
              key={d.driver_id}
              label={d.driver_name ?? `Driver ${d.driver_id}`}
              value={d.points}
              maxValue={maxDriverPoints}
              displayValue={`${d.points} pts`}
              color={constructorColor(teamByDriver.get(d.driver_id) ?? null)}
              index={i}
            />
          ))
        ) : (
          <p className="text-sm text-[var(--color-muted-text)]">Driver standings unavailable.</p>
        )}
      </div>
      <div className="flex flex-col gap-2">
        <p className="label-caps text-xs text-[var(--color-muted-text)]">Constructors</p>
        {constructors.length ? (
          constructors.slice(0, 10).map((c, i) => (
            <RankedBar
              key={c.constructor_id}
              label={c.constructor_name ?? `Constructor ${c.constructor_id}`}
              value={c.points}
              maxValue={maxConstructorPoints}
              displayValue={`${c.points} pts`}
              color={constructorColor(c.constructor_name)}
              index={i}
            />
          ))
        ) : (
          <p className="text-sm text-[var(--color-muted-text)]">Constructor standings unavailable.</p>
        )}
      </div>
    </div>
  );
}
