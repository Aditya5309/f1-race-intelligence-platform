/**
 * Ported from app/views/components.py (CONSTRUCTOR_COLORS,
 * DEFAULT_CONSTRUCTOR_COLOR, constructor_color()) — single source of truth
 * for the base set stays the Python module; keep in sync if that file
 * changes. Substring-tolerant matching preserved so renames like
 * "RB F1 Team" still resolve.
 *
 * DIVERGES from the Python source in one place: "audi" and "cadillac"
 * were added here (frontend only) because live 2026 predictions data
 * returns constructor_name "Audi" / "Cadillac" (Sauber's 2026 rebrand and
 * the new grid entrant) and the Python dict predates both, so it falls
 * back to DEFAULT_CONSTRUCTOR_COLOR for them — same gap exists upstream in
 * app/views/components.py, out of scope to fix here (Streamlit is off
 * limits this phase). Colors are representative, not exact brand hex
 * (matching this file's own existing precedent, e.g. "toyota": "#CC0000")
 * — sourced from 2026 livery-reveal coverage: Audi's is titanium/silver
 * with "Audi Red" + black accents (motor1.com, motorsportweek.com);
 * Cadillac's is a black-to-white gradient, no gold (formula1.com,
 * news.gm.com) — confirmed via live web search, not guessed.
 */

const OTHER_COLOR = "#4C78A8";

export const CONSTRUCTOR_COLORS: Record<string, string> = {
  "red bull": "#3671C6",
  ferrari: "#E8002D",
  mercedes: "#27F4D2",
  mclaren: "#FF8000",
  alpine: "#FF87BC",
  "aston martin": "#229971",
  williams: "#64C4FF",
  haas: "#B6BABD",
  alphatauri: "#6692FF",
  "rb f1": "#6692FF",
  "toro rosso": "#2B4562",
  "alfa romeo": "#C92D4B",
  sauber: "#52E252",
  renault: "#FFF500",
  "racing point": "#F596C8",
  "force india": "#F596C8",
  lotus: "#B8A15A",
  brawn: "#B8FD6E",
  caterham: "#048646",
  marussia: "#6E0000",
  manor: "#323232",
  jaguar: "#0A5C2F",
  toyota: "#CC0000",
  bmw: "#0066B2",
  // --- frontend-only additions, see divergence note above ---
  audi: "#BB0A30",
  cadillac: "#1C1C1C",
};

export const DEFAULT_CONSTRUCTOR_COLOR = OTHER_COLOR;

/** Brand hex for a constructor display name; neutral fallback. */
export function constructorColor(name: string | null | undefined): string {
  if (!name) return DEFAULT_CONSTRUCTOR_COLOR;
  const low = name.toLowerCase();
  for (const [key, color] of Object.entries(CONSTRUCTOR_COLORS)) {
    if (low.includes(key)) return color;
  }
  return DEFAULT_CONSTRUCTOR_COLOR;
}
