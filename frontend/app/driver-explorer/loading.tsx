import { NavShell } from "@/components/NavShell";
import {
  SkeletonChart,
  SkeletonEntityProfile,
  SkeletonStatRow,
  SkeletonTable,
  SkeletonTrendChart,
} from "@/components/skeleton";

export default function Loading() {
  return (
    <>
      <NavShell />
      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-4 px-6 py-12">
        <div>
          <h1 className="font-[family-name:var(--font-display)] text-3xl font-semibold">
            Driver Explorer
          </h1>
          <p className="mt-1 text-sm text-[var(--color-muted-text)]">
            A driver&apos;s races, one season or a whole career.
          </p>
        </div>

        <div className="flex flex-col gap-8">
          <SkeletonEntityProfile label="Loading driver profile…" />
          <SkeletonStatRow columns={6} label="Loading career stats…" />
          <SkeletonStatRow columns={4} label="Loading scope stats…" />

          <section>
            <p className="label-caps text-xs text-[var(--color-muted-text)]">Trends</p>
            <div className="mt-3 grid gap-6 sm:grid-cols-2">
              <SkeletonTrendChart label="Loading qualifying trend…" />
              <SkeletonTrendChart label="Loading finishing trend…" />
            </div>
            <div className="mt-6 grid gap-6 sm:grid-cols-2">
              <SkeletonTrendChart label="Loading points progression…" />
              <SkeletonTrendChart label="Loading win probability trend…" />
            </div>
          </section>

          <section>
            <p className="label-caps text-xs text-[var(--color-muted-text)]">Season profile</p>
            <div className="mt-3">
              <SkeletonChart height="h-72" variant="radar" label="Loading season profile…" />
            </div>
          </section>

          <section>
            <p className="label-caps text-xs text-[var(--color-muted-text)]">Race log</p>
            <div className="mt-3">
              <SkeletonTable rows={6} columns={5} label="Loading race log…" />
            </div>
          </section>
        </div>
      </main>
    </>
  );
}
