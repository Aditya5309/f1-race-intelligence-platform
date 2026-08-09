import { NavShell } from "@/components/NavShell";
import { SkeletonChart, SkeletonList, SkeletonStatRow } from "@/components/skeleton";

export default function Loading() {
  return (
    <>
      <NavShell />
      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-10 px-6 py-12">
        <div>
          <h1 className="font-[family-name:var(--font-display)] text-3xl font-semibold">
            Season Analytics
          </h1>
          <p className="mt-1 text-sm text-[var(--color-muted-text)]">
            How the season is unfolding, round by round.
          </p>
        </div>

        <SkeletonStatRow columns={3} label="Loading season stats…" />

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">
            Model accuracy over the season
          </p>
          <div className="mt-3">
            <SkeletonChart label="Loading accuracy chart…" />
          </div>
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Championship standings</p>
          <div className="mt-3 grid gap-8 sm:grid-cols-2">
            <SkeletonList items={5} label="Loading driver standings…" />
            <SkeletonList items={5} label="Loading constructor standings…" />
          </div>
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Most surprising races</p>
          <div className="mt-3">
            <SkeletonList variant="grid" items={3} label="Loading surprising races…" />
          </div>
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Win-share distribution</p>
          <div className="mt-3">
            <SkeletonChart variant="bars" label="Loading win-share distribution…" />
          </div>
        </section>
      </main>
    </>
  );
}
