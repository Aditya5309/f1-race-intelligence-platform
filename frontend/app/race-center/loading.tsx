import { NavShell } from "@/components/NavShell";
import { SkeletonHero, SkeletonList, SkeletonTable } from "@/components/skeleton";

export default function Loading() {
  return (
    <>
      <NavShell />
      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-4 px-6 py-12">
        <div>
          <h1 className="font-[family-name:var(--font-display)] text-3xl font-semibold">
            Race Center
          </h1>
          <p className="mt-1 text-sm text-[var(--color-muted-text)]">
            Pick a race — see who the model backs, and who could spoil it.
          </p>
        </div>

        <div className="flex flex-col gap-8">
          <SkeletonHero variant="feature" label="Loading race prediction…" />
          <SkeletonList variant="grid" items={3} label="Loading predicted podium…" />
          <SkeletonTable rows={10} columns={4} hasHeader={false} label="Loading full field…" />
        </div>
      </main>
    </>
  );
}
