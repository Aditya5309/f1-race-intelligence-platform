import { NavShell } from "@/components/NavShell";
import { SkeletonEntityProfile, SkeletonStatRow, SkeletonTrendChart } from "@/components/skeleton";

export default function Loading() {
  return (
    <>
      <NavShell />
      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-4 px-6 py-12">
        <div>
          <h1 className="font-[family-name:var(--font-display)] text-3xl font-semibold">Team</h1>
          <p className="mt-1 text-sm text-[var(--color-muted-text)]">
            A constructor&apos;s season or career — roster, form, and model accuracy.
          </p>
        </div>

        <div className="flex flex-col gap-8">
          <SkeletonEntityProfile label="Loading team profile…" />
          <SkeletonStatRow columns={5} label="Loading team stats…" />

          <section>
            <p className="label-caps text-xs text-[var(--color-muted-text)]">Trends</p>
            <div className="mt-3 grid gap-6 sm:grid-cols-2">
              <SkeletonTrendChart label="Loading qualifying trend…" />
              <SkeletonTrendChart label="Loading finishing trend…" />
            </div>
          </section>
        </div>
      </main>
    </>
  );
}
