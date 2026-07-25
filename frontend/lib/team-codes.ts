/**
 * Constructor display name -> 3-letter code, for the contender-card pill
 * badge. Frontend-only addition — no equivalent exists in the Python
 * source to port (Streamlit shows the full team name + a colored emoji
 * dot instead, see app/views/components.py's constructor_dot()). Driver
 * nationality isn't available over HTTP anywhere (see the Phase 2 plan's
 * "real data-availability gap" note), so this shows team code, not
 * country, despite the original Pit Wall reference using a country pill.
 *
 * Same lowercase-substring matching convention as constructor-colors.ts,
 * kept as a separate lookup since codes and colors change independently.
 */
const TEAM_CODES: Record<string, string> = {
  "red bull": "RBR",
  ferrari: "FER",
  mercedes: "MER",
  mclaren: "MCL",
  alpine: "ALP",
  "aston martin": "AMR",
  williams: "WIL",
  haas: "HAA",
  alphatauri: "AT",
  "rb f1": "RB",
  "toro rosso": "STR",
  "alfa romeo": "ALF",
  sauber: "SAU",
  renault: "REN",
  "racing point": "RP",
  "force india": "FI",
  lotus: "LOT",
  brawn: "BGP",
  caterham: "CAT",
  marussia: "MRS",
  manor: "MAN",
  jaguar: "JAG",
  toyota: "TOY",
  bmw: "BMW",
  audi: "AUD",
  cadillac: "CAD",
};

const DEFAULT_TEAM_CODE = "—";

export function teamCode(name: string | null | undefined): string {
  if (!name) return DEFAULT_TEAM_CODE;
  const low = name.toLowerCase();
  for (const [key, code] of Object.entries(TEAM_CODES)) {
    if (low.includes(key)) return code;
  }
  return DEFAULT_TEAM_CODE;
}
