/**
 * Ported verbatim from app/views/insights.py's hardcoded validation-results
 * stat row and model-selection paragraph — these figures are recorded at
 * promotion time, not recomputed live for whatever version is currently
 * serving (see that file's own comment). Keep in sync if it changes.
 */
const STATS = [
  { label: "Top-1 · validation 2022–23", value: "68.2%", help: "Pole-sitter baseline on the same races: 54.5%" },
  { label: "Top-3 recall · validation", value: "90.9%" },
  {
    label: "Top-1 · final test 2024",
    value: "45.8%",
    help: "Equal to the pole baseline — the edge is dominance-season concentrated",
  },
  {
    label: "ECE after calibration",
    value: "0.011",
    help: "Expected calibration error on validation, isotonic calibration fit on out-of-fold training predictions",
  },
];

/** Near-black hero treatment (mirrors RaceCenterHero / ModelStatusCard) —
 * the roadmap's design mapping for Model Insights' headline metrics. */
export function ValidationStats() {
  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-2xl bg-[var(--color-feature-card-bg)] p-6 text-[var(--color-text-on-dark)]">
        <div className="grid gap-4 sm:grid-cols-4">
          {STATS.map((s) => (
            <div key={s.label} title={s.help}>
              <p className="tabular text-2xl font-semibold">{s.value}</p>
              <p className="label-caps text-[10px] text-[var(--color-muted-text)]">{s.label}</p>
            </div>
          ))}
        </div>
      </div>
      <p className="text-sm text-[var(--color-muted-text)]">
        The serving model is a <strong>calibrated logistic regression</strong> chosen over
        random forest, XGBoost and LightGBM on validation performance, consistency, and
        simplicity. Full evidence: <code>reports/model_selection_report.md</code>.
      </p>
    </div>
  );
}
