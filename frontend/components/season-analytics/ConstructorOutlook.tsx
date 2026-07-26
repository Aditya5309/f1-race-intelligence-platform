import { constructorColor } from "@/lib/constructor-colors";
import type { ConstructorOutlookRow } from "@/lib/season-aggregation";
import { ProbabilityBar } from "@/components/race-center/ProbabilityBar";

export function ConstructorOutlook({ rows }: { rows: ConstructorOutlookRow[] }) {
  if (!rows.length) {
    return (
      <p className="text-sm text-[var(--color-muted-text)]">
        Constructor names unavailable for this season.
      </p>
    );
  }
  return (
    <div className="flex flex-col gap-2">
      {rows.map((r, i) => (
        <div key={r.constructor_name} className="flex items-center gap-3">
          <p className="w-28 shrink-0 truncate text-sm sm:w-36">{r.constructor_name}</p>
          <div className="flex-1">
            <ProbabilityBar
              probability={r.win_probability}
              color={constructorColor(r.constructor_name)}
              delay={i * 0.06}
            />
          </div>
          <p className="tabular w-14 shrink-0 text-right text-xs text-[var(--color-muted-text)]">
            {(r.win_probability * 100).toFixed(1)}%
          </p>
        </div>
      ))}
    </div>
  );
}
