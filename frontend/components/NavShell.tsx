"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";

const NAV_LINKS = [
  { href: "/race-center", label: "Race Center" },
  { href: "/season-analytics", label: "Season Analytics" },
  { href: "/model-insights", label: "Model Insights" },
  { href: "/driver-explorer", label: "Driver Explorer" },
  { href: "/compare-drivers", label: "Compare Drivers" },
  { href: "/team", label: "Team" },
  { href: "/circuit-explorer", label: "Circuit Explorer" },
];

/**
 * Below `md`, the inline link row is replaced by a toggled fixed-position
 * overlay (Task A1 of .ai/frontend_stabilization_roadmap.md) reusing the
 * same NAV_LINKS list. The overlay covers the viewport below the header
 * (the header stays above it via z-index, keeping the close control
 * reachable) rather than pushing page content down. No entrance/exit
 * animation on the overlay itself, matching lib/design-tokens.ts's
 * static-chrome convention for this component — only the hamburger icon's
 * bars use a small CSS transition to morph into an X. Client component only
 * because of the open/close toggle; the desktop-width row renders
 * identically to before.
 */
export function NavShell() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [lastPathname, setLastPathname] = useState(pathname);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLElement>(null);
  const firstLinkRef = useRef<HTMLAnchorElement>(null);

  // Close on route change — adjusting state during render on a prop change,
  // not in an effect (React's own recommended pattern; avoids the
  // cascading-render lint rule a naive `useEffect(() => setOpen(false),
  // [pathname])` would trigger). Kept as-is on review: functionally
  // equivalent to the roadmap's literal wording and avoids a real lint
  // error the literal version triggers.
  if (pathname !== lastPathname) {
    setLastPathname(pathname);
    setOpen(false);
  }

  // While open: Escape closes (returning focus to the trigger); Tab/Shift+Tab
  // is trapped inside the trigger + link list so focus can't escape to the
  // page behind the overlay; a mousedown outside both the menu and the
  // trigger also closes it.
  useEffect(() => {
    if (!open) return;

    function focusable(): HTMLElement[] {
      const links = menuRef.current
        ? Array.from(menuRef.current.querySelectorAll<HTMLElement>("a[href]"))
        : [];
      return triggerRef.current ? [triggerRef.current, ...links] : links;
    }

    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        setOpen(false);
        triggerRef.current?.focus();
        return;
      }
      if (e.key !== "Tab") return;
      const els = focusable();
      if (els.length === 0) return;
      const first = els[0];
      const last = els[els.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }

    function onMouseDown(e: MouseEvent) {
      const target = e.target as Node;
      if (menuRef.current?.contains(target) || triggerRef.current?.contains(target)) {
        return;
      }
      setOpen(false);
    }

    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("mousedown", onMouseDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("mousedown", onMouseDown);
    };
  }, [open]);

  // Move focus into the menu when it opens.
  useEffect(() => {
    if (open) firstLinkRef.current?.focus();
  }, [open]);

  return (
    <header className="relative z-50 border-b border-[var(--color-hairline)]">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-5">
        <Link href="/" className="label-caps text-sm text-[var(--color-text)]">
          Pit&nbsp;Wall&nbsp;/&nbsp;F1 Predictions
        </Link>

        <nav className="hidden flex-wrap gap-6 md:flex">
          {NAV_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="label-caps text-xs text-[var(--color-text)]"
            >
              {link.label}
            </Link>
          ))}
        </nav>

        <button
          ref={triggerRef}
          type="button"
          aria-expanded={open}
          aria-controls="mobile-nav-menu"
          aria-label={open ? "Close navigation menu" : "Open navigation menu"}
          onClick={() => setOpen((v) => !v)}
          className="flex h-8 w-8 flex-col items-center justify-center gap-1.5 md:hidden"
        >
          <span
            className={`h-px w-5 bg-[var(--color-text)] transition-transform ${open ? "translate-y-2 rotate-45" : ""}`}
          />
          <span
            className={`h-px w-5 bg-[var(--color-text)] transition-opacity ${open ? "opacity-0" : ""}`}
          />
          <span
            className={`h-px w-5 bg-[var(--color-text)] transition-transform ${open ? "-translate-y-2 -rotate-45" : ""}`}
          />
        </button>
      </div>

      {open && (
        <nav
          id="mobile-nav-menu"
          ref={menuRef}
          className="fixed inset-0 z-40 flex flex-col gap-1 bg-[var(--color-page-bg)] px-6 pt-24 pb-6 md:hidden"
        >
          {NAV_LINKS.map((link, i) => (
            <Link
              key={link.href}
              href={link.href}
              ref={i === 0 ? firstLinkRef : undefined}
              className="label-caps py-2 text-sm text-[var(--color-text)]"
            >
              {link.label}
            </Link>
          ))}
        </nav>
      )}
    </header>
  );
}
