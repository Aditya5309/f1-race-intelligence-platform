import { NavShell } from "@/components/NavShell";
import { FeatureClasses } from "@/components/model-insights/FeatureClasses";
import { FeatureImportanceChart } from "@/components/model-insights/FeatureImportanceChart";
import { ModelCard } from "@/components/model-insights/ModelCard";
import { ModelComparisonTable } from "@/components/model-insights/ModelComparisonTable";
import { ShapBarChart } from "@/components/model-insights/ShapBarChart";
import { ValidationStats } from "@/components/model-insights/ValidationStats";
import {
  getFeatureImportance,
  getModel,
  getModelComparison,
  getShapSummary,
  type ModelInfo,
} from "@/lib/api";

export default async function ModelInsightsPage() {
  const model: ModelInfo | null = await getModel().catch(() => null);
  const comparison = await getModelComparison().catch(() => ({ columns: [], rows: [] }));
  const importance = await getFeatureImportance().catch(() => ({ features: [], by_class: [] }));
  const shap = await getShapSummary().catch(() => ({ features: [] }));

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

        {model && (
          <section>
            <p className="label-caps text-xs text-[var(--color-muted-text)]">Model card</p>
            <div className="mt-3">
              <ModelCard model={model} />
            </div>
          </section>
        )}

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Validation results</p>
          <p className="mt-1 text-xs text-[var(--color-muted-text)]">
            Figures below are recorded at promotion time, not recomputed live for whatever
            version is currently serving.
          </p>
          <div className="mt-3">
            <ValidationStats />
          </div>
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Model comparison</p>
          <div className="mt-3">
            <ModelComparisonTable table={comparison} />
          </div>
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Feature classes</p>
          <div className="mt-3">
            <FeatureClasses />
          </div>
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">Feature importance</p>
          <div className="mt-3">
            <FeatureImportanceChart features={importance.features} byClass={importance.by_class} />
          </div>
        </section>

        <section>
          <p className="label-caps text-xs text-[var(--color-muted-text)]">SHAP analysis</p>
          <p className="mt-1 text-xs text-[var(--color-muted-text)]">Mean |SHAP| ranking.</p>
          <div className="mt-3">
            <ShapBarChart features={shap.features} />
          </div>
          <p className="mt-3 text-xs text-[var(--color-muted-text)]">
            The SHAP beeswarm, per-race waterfall case studies, calibration reliability
            diagram, confusion matrix, ROC curve, and precision-recall curve aren&apos;t
            available over HTTP yet — they have no underlying tabular data anywhere in this
            project&apos;s own pipeline (see <code>scripts/export_analysis_data.py</code>).
            View them locally via <code>reports/phase4_analysis/</code> or the Streamlit
            dashboard&apos;s Model Insights page in the meantime.
          </p>
        </section>
      </main>
    </>
  );
}
