import { NavShell } from "@/components/NavShell";
import { SkeletonCard, SkeletonChart, SkeletonStatRow, SkeletonTable } from "@/components/skeleton";

export default function Loading() {
  return (
    <>
      <NavShell />
      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-4 px-6 py-12">
        <div>
          <h1 className="font-[family-name:var(--font-display)] text-3xl font-semibold">
            Circuit Explorer
          </h1>
          <p className="mt-1 text-sm text-[var(--color-muted-text)]">
            One circuit&apos;s history — winners, risk, and pace over time.
          </p>
        </div>

        <div className="flex flex-col gap-6">
          <SkeletonChart height="h-72" label="Loading track outline…" />
          <SkeletonStatRow columns={3} label="Loading circuit stats…" />

          <section>
            <p className="label-caps text-xs text-[var(--color-muted-text)]">
              Most successful drivers here
            </p>
            <div className="mt-3">
              <SkeletonCard lines={5} label="Loading most successful drivers…" />
            </div>
          </section>

          <SkeletonTable rows={6} columns={3} label="Loading winners by year…" />
        </div>
      </main>
    </>
  );
}
