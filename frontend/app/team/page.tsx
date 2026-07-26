import { NavShell } from "@/components/NavShell";
import { TeamView } from "@/components/team/TeamView";
import { listRaces, type RaceSummary } from "@/lib/api";

export default async function TeamPage() {
  const races: RaceSummary[] = await listRaces().catch(() => []);
  const years = Array.from(new Set(races.map((r) => r.year))).sort((a, b) => a - b);

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

        {years.length === 0 ? (
          <p className="text-sm text-[var(--color-muted-text)]">No races available from the API.</p>
        ) : (
          <TeamView years={years} />
        )}
      </main>
    </>
  );
}
