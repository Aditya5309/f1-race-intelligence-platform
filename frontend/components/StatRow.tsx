/** Shared stat-tile row — used by Season Analytics, Driver Explorer, Team,
 * and Circuit Explorer. `columns` defaults to 3; pass more for pages with
 * more tiles. Reflows responsively (2 columns below `sm`, 3 at `sm`, the
 * full count at `lg`) via a pre-written class map — Tailwind can't
 * interpolate an arbitrary `columns` value into a class string at build
 * time, so every value actually used in this codebase needs its own
 * literal entry here. */
const COLUMN_CLASSES: Record<3 | 4 | 5 | 6, string> = {
  3: "grid-cols-2 sm:grid-cols-3 lg:grid-cols-3",
  4: "grid-cols-2 sm:grid-cols-3 lg:grid-cols-4",
  5: "grid-cols-2 sm:grid-cols-3 lg:grid-cols-5",
  6: "grid-cols-2 sm:grid-cols-3 lg:grid-cols-6",
};

export function StatRow({
  stats,
  columns = 3,
}: {
  stats: { label: string; value: string; help?: string }[];
  columns?: 3 | 4 | 5 | 6;
}) {
  return (
    <div className={`grid gap-4 ${COLUMN_CLASSES[columns]}`}>
      {stats.map((s) => (
        <div key={s.label} className="rounded-xl bg-white/40 p-4" title={s.help}>
          <p className="tabular text-2xl font-semibold">{s.value}</p>
          <p className="label-caps text-[10px] text-[var(--color-muted-text)]">{s.label}</p>
        </div>
      ))}
    </div>
  );
}
