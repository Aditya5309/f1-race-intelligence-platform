import Link from "next/link";

/**
 * Static chrome — deliberately NOT animated (see lib/design-tokens.ts's
 * motion token comments). Links to the other six Streamlit-dashboard page
 * equivalents are still inert placeholders (Phase 2 built Race Center only).
 */
const PLACEHOLDER_PAGES = [
  "Driver Explorer",
  "Compare Drivers",
  "Team",
  "Circuit Explorer",
  "Season Analytics",
  "Model Insights",
];

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
