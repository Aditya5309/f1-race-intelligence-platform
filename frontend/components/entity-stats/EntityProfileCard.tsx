/** Shared profile-card shell — Driver Explorer's driver bio and Team's
 * constructor bio (mirrors both pages' st.container(border=True) profile
 * card). Not the near-black hero treatment (that's reserved for Race
 * Center's/Model Insights' headline metrics per the roadmap's own design
 * mapping) — a bordered secondary card, matching this page's own scope. */
export function EntityProfileCard({
  color,
  name,
  subtitle,
  championshipLine,
}: {
  color: string;
  name: string;
  subtitle: string;
  championshipLine?: string | null;
}) {
  return (
    <div className="rounded-2xl border border-[var(--color-hairline)] bg-white/40 p-6">
      <div className="flex items-start gap-4">
        <span
          aria-hidden
          className="mt-1.5 h-4 w-4 shrink-0 rounded-full"
          style={{ backgroundColor: color }}
        />
        <div>
          <h2 className="font-[family-name:var(--font-display)] text-2xl font-semibold">{name}</h2>
          <p className="mt-1 text-sm text-[var(--color-muted-text)]">{subtitle}</p>
          {championshipLine && <p className="mt-2 text-sm">🏆 {championshipLine}</p>}
        </div>
      </div>
    </div>
  );
}
