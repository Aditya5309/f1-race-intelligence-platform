/**
 * Track outline from OpenStreetMap (scripts/backfill_circuit_layouts.py).
 * A lightweight, dependency-free SVG path — not a Recharts chart, since
 * this needs a locked-aspect-ratio geographic projection, not a data axis.
 * Coordinates are projected to a local equirectangular meter grid (lon
 * scaled by cos(latitude)) before plotting, mirroring
 * app/views/charts.py::circuit_layout_map exactly — an uncorrected
 * lat/lon plot would distort every circuit except ones on the equator.
 */
const METERS_PER_DEGREE = 111_320;

export function TrackOutline({
  coordinates,
  color = "#4C78A8",
}: {
  coordinates: [number, number][];
  color?: string;
}) {
  if (!coordinates.length) return null;

  const lats = coordinates.map((c) => c[1]);
  const meanLatRad = (lats.reduce((a, b) => a + b, 0) / lats.length) * (Math.PI / 180);
  const points = coordinates.map(([lon, lat]) => [
    lon * METERS_PER_DEGREE * Math.cos(meanLatRad),
    -lat * METERS_PER_DEGREE, // flip Y: SVG y grows downward, latitude grows upward
  ]);

  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const span = Math.max(maxX - minX, maxY - minY) || 1;
  const padding = span * 0.08;
  const viewBox = `${minX - padding} ${minY - padding} ${maxX - minX + 2 * padding} ${maxY - minY + 2 * padding}`;
  const pathD = points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x} ${y}`).join(" ");

  return (
    <svg viewBox={viewBox} className="h-full w-full" preserveAspectRatio="xMidYMid meet">
      <path
        d={pathD}
        fill="none"
        stroke={color}
        strokeWidth={span * 0.012}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
