export function StatRow({
  stats,
}: {
  stats: { label: string; value: string; help?: string }[];
}) {
  return (
    <div className="grid grid-cols-3 gap-4">
      {stats.map((s) => (
        <div key={s.label} className="rounded-xl bg-white/40 p-4" title={s.help}>
          <p className="tabular text-2xl font-semibold">{s.value}</p>
          <p className="label-caps text-[10px] text-[var(--color-muted-text)]">{s.label}</p>
        </div>
      ))}
    </div>
  );
}
