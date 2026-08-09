import { NavShell } from "@/components/NavShell";
import { Skeleton, SkeletonChart, SkeletonEntityProfile } from "@/components/skeleton";

export default function Loading() {
  return (
    <>
      <NavShell />
      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-4 px-6 py-12">
        <div>
          <h1 className="font-[family-name:var(--font-display)] text-3xl font-semibold">
            Compare Drivers
          </h1>
          <p className="mt-1 text-sm text-[var(--color-muted-text)]">
            Two drivers, one season — side by side.
          </p>
        </div>

        <div className="flex flex-col gap-8">
          <div className="grid gap-6 sm:grid-cols-2">
            <SkeletonEntityProfile label="Loading first driver…" />
            <SkeletonEntityProfile label="Loading second driver…" />
          </div>

          <section>
            <p className="label-caps text-xs text-[var(--color-muted-text)]">
              Predicted win share for a race this season
            </p>
            <div className="mt-3 grid gap-4 sm:grid-cols-2">
              {[0, 1].map((i) => (
                <div key={i} role="status" aria-live="polite" className="rounded-xl bg-white/40 p-4">
                  <span className="sr-only">Loading win share…</span>
                  <Skeleton className="h-4 w-1/2" />
                  <Skeleton className="mt-2 h-7 w-1/3" />
                </div>
              ))}
            </div>
          </section>

          <section>
            <p className="label-caps text-xs text-[var(--color-muted-text)]">Season profile</p>
            <div className="mt-3">
              <SkeletonChart height="h-72" variant="radar" label="Loading season profile…" />
            </div>
          </section>
        </div>
      </main>
    </>
  );
}
