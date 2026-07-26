import type { ModelInfo } from "@/lib/api";

export function ModelCard({ model }: { model: ModelInfo }) {
  return (
    <div className="grid gap-4 sm:grid-cols-4">
      {[
        { label: "Algorithm", value: model.model_class },
        { label: "Calibration", value: model.calibration },
        { label: "Registry alias", value: `v${model.version} @ ${model.alias}` },
        { label: "Trained", value: model.trained_at.slice(0, 10) },
      ].map((s) => (
        <div key={s.label} className="rounded-xl bg-white/40 p-4">
          <p className="tabular text-lg font-semibold">{s.value}</p>
          <p className="label-caps text-[10px] text-[var(--color-muted-text)]">{s.label}</p>
        </div>
      ))}
      <p className="col-span-full text-xs text-[var(--color-muted-text)]">
        Run id: <span className="tabular">{model.run_id}</span>
      </p>
    </div>
  );
}
