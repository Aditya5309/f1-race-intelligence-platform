"use client";

import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { featureClassColor } from "@/lib/feature-classes";
import type { ShapSummaryRow } from "@/lib/api";

/** Native replacement for shap_bar_logreg.png's mean |SHAP| ranking. The
 * beeswarm/waterfall SHAP figures have no route (Decision 063) — see
 * scripts/export_analysis_data.py. */
export function ShapBarChart({ features }: { features: ShapSummaryRow[] }) {
  if (!features.length) {
    return (
      <p className="text-sm text-[var(--color-muted-text)]">
        SHAP summary data not found — run <code>python scripts/export_analysis_data.py</code>.
      </p>
    );
  }
  const top20 = features.slice(0, 20);
  return (
    <div style={{ height: Math.max(320, top20.length * 22) }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={top20} layout="vertical" margin={{ left: 24 }}>
          <CartesianGrid stroke="var(--color-hairline)" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 11 }} />
          <YAxis type="category" dataKey="feature" width={180} tick={{ fontSize: 10 }} />
          <Tooltip />
          <Bar dataKey="mean_abs_shap" radius={[0, 3, 3, 0]}>
            {top20.map((row) => (
              <Cell key={row.feature} fill={featureClassColor(row.feature_class)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
