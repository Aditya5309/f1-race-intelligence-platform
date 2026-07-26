import Link from "next/link";

/**
 * Static chrome — deliberately NOT animated (see lib/design-tokens.ts's
 * motion token comments). Every Streamlit-dashboard page equivalent is now
 * converted (Phase 6, Circuit Explorer, was the last one) — no placeholder
 * pages remain.
 */
export function NavShell() {
  return (
    <header className="border-b border-[var(--color-hairline)]">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-5">
        <Link href="/" className="label-caps text-sm text-[var(--color-text)]">
          Pit&nbsp;Wall&nbsp;/&nbsp;F1 Predictions
        </Link>
        <nav className="hidden flex-wrap gap-6 md:flex">
          <Link href="/race-center" className="label-caps text-xs text-[var(--color-text)]">
            Race Center
          </Link>
          <Link href="/season-analytics" className="label-caps text-xs text-[var(--color-text)]">
            Season Analytics
          </Link>
          <Link href="/model-insights" className="label-caps text-xs text-[var(--color-text)]">
            Model Insights
          </Link>
          <Link href="/driver-explorer" className="label-caps text-xs text-[var(--color-text)]">
            Driver Explorer
          </Link>
          <Link href="/compare-drivers" className="label-caps text-xs text-[var(--color-text)]">
            Compare Drivers
          </Link>
          <Link href="/team" className="label-caps text-xs text-[var(--color-text)]">
            Team
          </Link>
          <Link href="/circuit-explorer" className="label-caps text-xs text-[var(--color-text)]">
            Circuit Explorer
          </Link>
        </nav>
      </div>
    </header>
  );
}
