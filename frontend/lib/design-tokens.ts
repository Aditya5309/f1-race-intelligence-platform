/**
 * Design tokens for the F1 Race Winner Prediction frontend.
 *
 * Palette confirmed from f1.anirudhgoel.xyz screenshots (ground truth,
 * see .ai/frontend_overhaul_phase1_prompt.md): a warm editorial cream base
 * with near-black elevated feature cards — NOT a dark-mode-first design.
 * Team colors (constructor-colors.ts) are restricted to small accents
 * (border bars, pills) and must never be used as large fills.
 *
 * Single reusable source — components import from here, never redefine
 * a color/spacing value inline.
 */

export const color = {
  pageBackground: "#EFEAE0",
  featureCardBackground: "#0A0A0A",
  text: "#111111",
  textOnDark: "#F5F1E8",
  mutedText: "#8A8272", // warm gray, not cool gray
  accentRed: "#E23A2E", // italic emphasis words, live/urgent status
  accentBlue: "#3B6FE0", // CTAs, small highlight lines — sparing use only
  liveGreen: "#3FCB6B", // small live-status indicator dot only
  hairline: "#D9D2C2", // list-row dividers on the cream background
  hairlineOnDark: "#2A2A2A",
} as const;

export const font = {
  // Bold, high-contrast serif/Didone-style display font — italic used
  // specifically for emphasized words within a heading.
  display: "var(--font-display)",
  // Condensed uppercase small-caps for navigation/labels.
  label: "var(--font-label)",
  // Tabular monospace numerals for countdowns, points, probabilities.
  mono: "var(--font-mono)",
} as const;

export const spacing = {
  xs: "0.5rem",
  sm: "0.75rem",
  md: "1.5rem",
  lg: "2.5rem",
  xl: "4rem",
} as const;

export const motion = {
  // What animates: numbers ticking up, cards fading/sliding in on load,
  // smooth cross-fades between loading and loaded states.
  cardEnter: { duration: 0.5, ease: [0.16, 1, 0.3, 1] as const },
  numberTick: { duration: 0.6, ease: "easeOut" as const },
  staggerStep: 0.06,
  // What deliberately doesn't animate: static chrome (nav, footer), the
  // token colors themselves, anything that would refire on every
  // re-render of otherwise-unchanged data — see components for where
  // this boundary is enforced.
} as const;
