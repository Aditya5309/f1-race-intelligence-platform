"use client";

import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useChartReveal } from "@/components/motion";
import { featureClassColor } from "@/lib/feature-classes";
import type { FeatureImportanceRow, ImportanceByClassRow } from "@/lib/api";

/**
 * Native replacement for feature_importance_logreg.png (top 20) +
 * importance_by_class_logreg.png (same source data, summed by class) —
 * real charts, not screenshots (Decision 063).
 */
export function FeatureImportanceChart({
  features,
  byClass,
}: {
  features: FeatureImportanceRow[];
  byClass: ImportanceByClassRow[];
}) {
  const chartReveal = useChartReveal();
  if (!features.length) {
    return (
      <p className="text-sm text-[var(--color-muted-text)]">
        Feature importance data not found — run{" "}
        <code>python scripts/export_analysis_data.py</code>.
      </p>
    );
  }
  const top20 = features.slice(0, 20);

  return (
    <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
      <div>
        <p className="mb-2 text-xs text-[var(--color-muted-text)]">
          Native importance (|coefficient|), top 20 — grid/qualifying dominate
        </p>
        <div style={{ height: Math.max(320, top20.length * 22) }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={top20} layout="vertical" margin={{ left: 24 }}>
              <CartesianGrid stroke="var(--color-hairline)" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="feature" width={180} tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="importance" radius={[0, 3, 3, 0]} {...chartReveal}>
                {top20.map((row) => (
                  <Cell key={row.feature} fill={featureClassColor(row.feature_class)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div>
        <p className="mb-2 text-xs text-[var(--color-muted-text)]">Summed importance by feature class</p>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={byClass} layout="vertical" margin={{ left: 8 }}>
              <CartesianGrid stroke="var(--color-hairline)" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="feature_class" width={90} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="total_importance" radius={[0, 3, 3, 0]} {...chartReveal}>
                {byClass.map((row) => (
                  <Cell key={row.feature_class} fill={featureClassColor(row.feature_class)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
