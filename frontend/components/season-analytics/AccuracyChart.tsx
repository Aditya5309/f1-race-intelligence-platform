"use client";

import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { AccuracyPoint } from "@/lib/season-aggregation";

/** First real Recharts usage in this frontend (installed Phase 1 for
 * exactly this kind of need — no static PNG, no screenshot). */
export function AccuracyChart({ data }: { data: AccuracyPoint[] }) {
  if (!data.length) {
    return (
      <p className="text-sm text-[var(--color-muted-text)]">
        Not enough decided races yet this season.
      </p>
    );
  }
  return (
    <div className="grid gap-6 sm:grid-cols-2">
      <div>
        <p className="mb-2 text-xs text-[var(--color-muted-text)]">
          1 = the model&apos;s favorite won that round, 0 = it didn&apos;t.
        </p>
        <div className="h-52">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data}>
              <CartesianGrid stroke="var(--color-hairline)" vertical={false} />
              <XAxis dataKey="round" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 1]} ticks={[0, 1]} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="hit" fill="var(--color-accent-blue)" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div>
        <p className="mb-2 text-xs text-[var(--color-muted-text)]">Cumulative hit rate</p>
        <div className="h-52">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid stroke="var(--color-hairline)" vertical={false} />
              <XAxis dataKey="round" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(0)}%`} />
              <Line type="monotone" dataKey="cumulative" stroke="var(--color-accent-red)" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
