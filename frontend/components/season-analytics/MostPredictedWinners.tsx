import { RankedBar } from "@/components/RankedBar";
import { constructorColor } from "@/lib/constructor-colors";
import type { PickCount } from "@/lib/season-aggregation";

export function MostPredictedWinners({ picks }: { picks: PickCount[] }) {
  if (!picks.length) {
    return <p className="text-sm text-[var(--color-muted-text)]">No decided picks yet this season.</p>;
  }
  const maxPicks = Math.max(1, ...picks.map((p) => p.picks));
  return (
    <div className="flex flex-col gap-2">
      {picks.map((p, i) => (
        <RankedBar
          key={p.label}
          label={p.label}
          value={p.picks}
          maxValue={maxPicks}
          displayValue={`${p.picks}×`}
          color={constructorColor(p.team)}
          index={i}
        />
      ))}
    </div>
  );
}
