/** Shared stat-tile row — used by Season Analytics, Driver Explorer,
 * Compare Drivers, and Team. `columns` defaults to 3 (Season Analytics'
 * original, unchanged behavior); pass more for pages with more tiles. */
export function StatRow({
  stats,
  columns = 3,
}: {
  stats: { label: string; value: string; help?: string }[];
  columns?: number;
}) {
  return (
    <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>
      {stats.map((s) => (
        <div key={s.label} className="rounded-xl bg-white/40 p-4" title={s.help}>
          <p className="tabular text-2xl font-semibold">{s.value}</p>
          <p className="label-caps text-[10px] text-[var(--color-muted-text)]">{s.label}</p>
        </div>
      ))}
    </div>
  );
}
