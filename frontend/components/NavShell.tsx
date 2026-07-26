import Link from "next/link";

/**
 * Static chrome — deliberately NOT animated (see lib/design-tokens.ts's
 * motion token comments). Links to the remaining four Streamlit-dashboard
 * page equivalents are still inert placeholders (Phase 3 built Season
 * Analytics, Phase 4 built Model Insights; Driver Explorer/Compare
 * Drivers/Team/Circuit Explorer are still pending, per the roadmap's own
 * sequencing).
 */
const PLACEHOLDER_PAGES = ["Driver Explorer", "Compare Drivers", "Team", "Circuit Explorer"];

export function NavShell() {
  return (
    <header className="border-b border-[var(--color-hairline)]">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-5">
        <Link href="/" className="label-caps text-sm text-[var(--color-text)]">
          Pit&nbsp;Wall&nbsp;/&nbsp;F1 Predictions
        </Link>
        <nav className="hidden gap-6 md:flex">
          <Link href="/race-center" className="label-caps text-xs text-[var(--color-text)]">
            Race Center
          </Link>
          <Link href="/season-analytics" className="label-caps text-xs text-[var(--color-text)]">
            Season Analytics
          </Link>
          <Link href="/model-insights" className="label-caps text-xs text-[var(--color-text)]">
            Model Insights
          </Link>
          {PLACEHOLDER_PAGES.map((page) => (
            <span
              key={page}
              className="label-caps cursor-default text-xs text-[var(--color-muted-text)]"
              title="Coming in a later phase"
            >
              {page}
            </span>
          ))}
        </nav>
      </div>
    </header>
  );
}
