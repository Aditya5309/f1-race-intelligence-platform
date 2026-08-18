import type { ModelInfo } from "@/lib/api";

/** Near-black hero treatment (mirrors RaceCenterHero / ModelStatusCard) —
 * the roadmap's design mapping for Model Insights' headline metrics. */
export function ModelCard({ model }: { model: ModelInfo }) {
  return (
    <div className="rounded-2xl bg-[var(--color-feature-card-bg)] p-6 text-[var(--color-text-on-dark)]">
      <div className="grid gap-4 sm:grid-cols-4">
        {[
          { label: "Algorithm", value: model.model_class },
          { label: "Calibration", value: model.calibration },
          { label: "Registry alias", value: `v${model.version} @ ${model.alias}` },
          { label: "Trained", value: model.trained_at.slice(0, 10) },
        ].map((s) => (
          <div key={s.label}>
            <p className="tabular text-lg font-semibold">{s.value}</p>
            <p className="label-caps text-[10px] text-[var(--color-muted-text-on-dark)]">{s.label}</p>
          </div>
        ))}
      </div>
      <p className="mt-4 text-xs text-[var(--color-muted-text-on-dark)]">
        Run id: <span className="tabular">{model.run_id}</span>
      </p>
    </div>
  );
}
