import { NavShell } from "@/components/NavShell";
import { SkeletonCard, SkeletonHero, SkeletonList } from "@/components/skeleton";

export default function Loading() {
  return (
    <>
      <NavShell />
      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-10 px-6 py-12">
        <section className="grid gap-8 md:grid-cols-[2fr_1fr]">
          <SkeletonHero variant="bare" label="Loading race summary…" />
          <SkeletonCard variant="dark" lines={4} label="Loading model status…" />
        </section>

        <section>
          <SkeletonList items={5} label="Loading win probability…" />
        </section>
      </main>
      <footer className="border-t border-[var(--color-hairline)] px-6 py-6">
        <p className="label-caps text-center text-xs text-[var(--color-muted-text)]">
          F1 Race Winner Prediction &middot; Phase 1 pilot
        </p>
      </footer>
    </>
  );
}
