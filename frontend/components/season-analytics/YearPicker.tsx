"use client";

import { useRouter } from "next/navigation";

export function YearPicker({ years, selected }: { years: number[]; selected: number }) {
  const router = useRouter();
  return (
    <select
      value={selected}
      onChange={(e) => router.push(`/season-analytics?year=${e.target.value}`)}
      aria-label="Select season"
      className="rounded border border-[var(--color-hairline)] bg-transparent px-3 py-1.5 text-sm"
    >
      {years.map((y) => (
        <option key={y} value={y}>
          {y}
        </option>
      ))}
    </select>
  );
}
