import { NavShell } from "@/components/NavShell";
import { SkeletonCard, SkeletonChart, SkeletonStatRow, SkeletonTable } from "@/components/skeleton";

export default function Loading() {
  return (
    <>
      <NavShell />
      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-10 px-6 py-12">
        <div>
          <span className="label-caps rounded-full bg-[var(--color-feature-card-bg)] px-2.5 py-1 text-[10px] text-[var(--color-text-on-dark)]">
            Advanced / technical
          </span>
          <h1 className="mt-3 font-[family-name:var(--font-display)] text-3xl font-semibold">
            Model Insights
          </h1>
          <p className="mt-1 text-sm text-[var(--color-muted-text)]">
            Advanced — the ML internals behind the predictions.
          </p>
        </div>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Model card</p>
          <div className="mt-3">
            <SkeletonCard variant="dark" lines={4} label="Loading model card…" />
          </div>
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Validation results</p>
          <div className="mt-3">
            <SkeletonStatRow columns={4} label="Loading validation results…" />
          </div>
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Model comparison</p>
          <div className="mt-3">
            <SkeletonTable rows={5} columns={5} label="Loading model comparison…" />
          </div>
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Feature importance</p>
          <div className="mt-3">
            <SkeletonChart height="h-80" label="Loading feature importance…" />
          </div>
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">SHAP analysis</p>
          <div className="mt-3">
            <SkeletonChart height="h-80" label="Loading SHAP analysis…" />
          </div>
        </section>
      </main>
    </>
  );
}
