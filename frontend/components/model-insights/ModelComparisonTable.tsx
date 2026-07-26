import type { ModelComparisonResponse } from "@/lib/api";

function formatCell(value: number | string | boolean | null): string {
  if (value == null) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  return String(value);
}

export function ModelComparisonTable({ table }: { table: ModelComparisonResponse }) {
  if (!table.columns.length) {
    return (
      <p className="text-sm text-[var(--color-muted-text)]">
        model_comparison.csv not found — the comparison table lives in reports/ (local-only)
        and in MLflow (<code>mlflow ui</code>).
      </p>
    );
  }
  return (
    <div className="overflow-x-auto rounded-xl bg-white/40">
      <table className="w-full min-w-max text-left text-xs">
        <thead>
          <tr className="border-b border-[var(--color-hairline)]">
            {table.columns.map((col) => (
              <th key={col} className="label-caps px-3 py-2 font-normal text-[var(--color-muted-text)]">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, i) => (
            <tr key={i} className="border-b border-[var(--color-hairline)] last:border-none">
              {table.columns.map((col) => (
                <td key={col} className="tabular whitespace-nowrap px-3 py-2">
                  {formatCell(row[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
