import { NavShell } from "@/components/NavShell";
import { CircuitExplorerView } from "@/components/circuit-explorer/CircuitExplorerView";
import { getCircuits } from "@/lib/api";

export default async function CircuitExplorerPage() {
  const catalog = await getCircuits().catch(() => []);

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

        {catalog.length === 0 ? (
          <p className="text-sm text-[var(--color-muted-text)]">
            Circuit metadata unavailable (display data not loaded).
          </p>
        ) : (
          <CircuitExplorerView catalog={catalog} />
        )}
      </main>
    </>
  );
}
