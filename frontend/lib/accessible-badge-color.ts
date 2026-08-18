/**
 * Shared fix for the "category → hue → same hue for badge text and
 * background tint" pattern used by both FeatureClasses.tsx (feature-class
 * colors) and ContenderCard.tsx (constructor colors, via `${color}22`
 * backgrounds) — the literal hue as text color on its own ~13%-opacity
 * tint is contrast-locked by construction (Task C8). This preserves the
 * background tint and the hue's identity exactly as today, and darkens
 * only the text to a same-hue, same-saturation shade that reaches ≥4.6:1
 * (a small safety margin over WCAG AA's 4.5:1) against that tint.
 *
 * The tint's actual backdrop differs slightly by caller (FeatureClasses
 * sits on the page background, ContenderCard on a `bg-white/40` card) —
 * this always computes against the page background (the darker of the
 * two). A darker parent alpha-blends into a darker, lower-luminance tint,
 * which is the *harder* case to reach 4.5:1 against (contrast ratio rises
 * with the background's luminance, for a fixed darker text color) — so
 * passing against the page background guarantees passing against the
 * lighter card background too, without needing a caller-supplied one.
 */

const BADGE_TINT_ALPHA = 0x22 / 255;
const STRICTEST_BADGE_BACKDROP: RGB = [0xef, 0xea, 0xe0]; // page bg — darker than the card bg, the harder case
const TARGET_CONTRAST_RATIO = 4.6;

type RGB = [number, number, number];
type HSL = [number, number, number];

function hexToRgb(hex: string): RGB {
  const h = hex.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

function toHex([r, g, b]: RGB): string {
  return (
    "#" +
    [r, g, b]
      .map((c) => Math.round(Math.min(255, Math.max(0, c))).toString(16).padStart(2, "0"))
      .join("")
  );
}

function relativeLuminance([r, g, b]: RGB): number {
  const [R, G, B] = [r, g, b].map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * R + 0.7152 * G + 0.0722 * B;
}

function contrastRatio(a: RGB, b: RGB): number {
  const [L1, L2] = [relativeLuminance(a), relativeLuminance(b)].sort((x, y) => y - x);
  return (L1 + 0.05) / (L2 + 0.05);
}

function alphaBlend(fg: RGB, bg: RGB, alpha: number): RGB {
  return fg.map((c, i) => c * alpha + bg[i] * (1 - alpha)) as RGB;
}

function rgbToHsl([r, g, b]: RGB): HSL {
  const [rn, gn, bn] = [r / 255, g / 255, b / 255];
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  let h = 0;
  const l = (max + min) / 2;
  const d = max - min;
  const s = d === 0 ? 0 : l > 0.5 ? d / (2 - max - min) : d / (max + min);
  if (d !== 0) {
    switch (max) {
      case rn:
        h = (gn - bn) / d + (gn < bn ? 6 : 0);
        break;
      case gn:
        h = (bn - rn) / d + 2;
        break;
      default:
        h = (rn - gn) / d + 4;
    }
    h /= 6;
  }
  return [h, s, l];
}

function hslToRgb([h, s, l]: HSL): RGB {
  if (s === 0) {
    const v = l * 255;
    return [v, v, v];
  }
  const hue2rgb = (p: number, q: number, t: number) => {
    let tt = t;
    if (tt < 0) tt += 1;
    if (tt > 1) tt -= 1;
    if (tt < 1 / 6) return p + (q - p) * 6 * tt;
    if (tt < 1 / 2) return q;
    if (tt < 2 / 3) return p + (q - p) * (2 / 3 - tt) * 6;
    return p;
  };
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [255 * hue2rgb(p, q, h + 1 / 3), 255 * hue2rgb(p, q, h), 255 * hue2rgb(p, q, h - 1 / 3)];
}

/**
 * Returns a same-hue, same-saturation, darkened version of `hex` that
 * reaches ≥4.6:1 against its own `${hex}22`-style tinted background —
 * or `hex` unchanged if it already does (several brand colors, e.g. dark
 * navy/near-black team colors, already pass with no adjustment).
 */
export function accessibleBadgeTextColor(hex: string): string {
  const base = hexToRgb(hex);
  const tintBg = alphaBlend(base, STRICTEST_BADGE_BACKDROP, BADGE_TINT_ALPHA);
  const [h, s] = rgbToHsl(base);
  let l = rgbToHsl(base)[2];
  let candidate = base;
  let steps = 0;
  while (contrastRatio(candidate, tintBg) < TARGET_CONTRAST_RATIO && l > 0 && steps < 400) {
    l = Math.max(0, l - 0.005);
    candidate = hslToRgb([h, s, l]);
    steps++;
  }
  return toHex(candidate);
}
