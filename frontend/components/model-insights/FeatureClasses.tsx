import { ERA_SENSITIVE_FEATURES, EXPERIMENTAL_FEATURES, STABLE_FEATURES, featureClassColor } from "@/lib/feature-classes";
import { accessibleBadgeTextColor } from "@/lib/accessible-badge-color";

function Column({ title, features, color }: { title: string; features: string[]; color: string }) {
  const textColor = accessibleBadgeTextColor(color);
  return (
    <div>
      <p className="text-sm font-medium">
        {title} ({features.length})
      </p>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {features.map((f) => (
          <span
            key={f}
            className="rounded-full px-2 py-0.5 text-[10px] ring-1 ring-inset ring-black/15"
            style={{ backgroundColor: `${color}22`, color: textColor }}
          >
            {f}
          </span>
        ))}
      </div>
    </div>
  );
}

/** Ported from app/views/insights.py's feature-classes section, verbatim
 * copy and thresholds — see lib/feature-classes.ts. */
export function FeatureClasses() {
  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm text-[var(--color-muted-text)]">
        The original 31 model features, classified by era-robustness. A healthy model
        concentrates its signal in Stable features — this one draws ~59% of its importance
        from them, led by grid position and qualifying results. The currently-serving model
        may use additional features not yet classified here.
      </p>
      <div className="grid gap-6 sm:grid-cols-3">
        <Column title="Stable" features={STABLE_FEATURES} color={featureClassColor("stable")} />
        <Column
          title="Era-sensitive"
          features={ERA_SENSITIVE_FEATURES}
          color={featureClassColor("era_sensitive")}
        />
        <Column
          title="Experimental"
          features={EXPERIMENTAL_FEATURES}
          color={featureClassColor("experimental")}
        />
      </div>
    </div>
  );
}
