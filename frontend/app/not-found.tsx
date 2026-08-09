import Link from "next/link";
import { NavShell } from "@/components/NavShell";

export default function NotFound() {
  return (
    <>
      <NavShell />
      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-4 px-6 py-12">
        <div>
          <h1 className="font-[family-name:var(--font-display)] text-3xl font-semibold">
            🏁 Off the racing line
          </h1>
          <p className="mt-1 text-sm text-[var(--color-muted-text)]">
            This page doesn&apos;t exist — check the URL, or use the nav above.
          </p>
        </div>

        <Link href="/" className="label-caps text-xs text-[var(--color-accent-blue)]">
          Back to Home
        </Link>
      </main>
    </>
  );
}
