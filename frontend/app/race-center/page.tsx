import { NavShell } from "@/components/NavShell";
import { RaceCenterView } from "@/components/race-center/RaceCenterView";
import { getUpcomingRace, listRaces, type RaceSummary } from "@/lib/api";

export default async function RaceCenterPage({
  searchParams,
}: {
  searchParams: Promise<{ race_id?: string }>;
}) {
  const races: RaceSummary[] = await listRaces().catch(() => []);
  const upcoming = await getUpcomingRace().catch(() => null);

  // Default landing race = the latest COMPLETED race, computed BEFORE the
  // upcoming race is considered — mirrors app/views/race_center.py's own
  // ordering so a shared ?race_id= link behaves the same way in both apps.
  const historicalSorted = [...races].sort(
    (a, b) => a.year - b.year || a.round - b.round,
  );
  const defaultRaceId = historicalSorted[historicalSorted.length - 1]?.race_id ?? upcoming?.race_id;

  const params = await searchParams;
  const requestedId = params.race_id ? Number(params.race_id) : undefined;
  const validIds = new Set([...races.map((r) => r.race_id), upcoming?.race_id].filter(Boolean));
  const initialRaceId =
    requestedId != null && validIds.has(requestedId) ? requestedId : defaultRaceId;

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

        {initialRaceId == null ? (
          <p className="text-sm text-[var(--color-muted-text)]">No races available from the API.</p>
        ) : (
          <RaceCenterView races={races} upcoming={upcoming} initialRaceId={initialRaceId} />
        )}
      </main>
    </>
  );
}
