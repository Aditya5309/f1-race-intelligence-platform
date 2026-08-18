"use client";

import { FadeIn } from "@/components/motion";
import type { HealthResponse } from "@/lib/api";

export function ModelStatusCard({ health }: { health: HealthResponse | null }) {
  const ok = health?.status === "ok";
  const model = health?.model;

  return (
    <FadeIn className="rounded-2xl bg-[var(--color-feature-card-bg)] p-6 text-[var(--color-text-on-dark)]">
      <div className="flex items-center gap-2">
        <span
          className="h-2 w-2 rounded-full"
          style={{ backgroundColor: ok ? "var(--color-live-green)" : "var(--color-accent-red)" }}
        />
        <span className="label-caps text-xs text-[var(--color-muted-text-on-dark)]">
          {ok ? "Model live" : "Model unavailable"}
        </span>
      </div>

      {model ? (
        <dl className="mt-4 grid grid-cols-2 gap-y-3 text-sm">
          <dt className="label-caps text-[var(--color-muted-text-on-dark)]">Alias</dt>
          <dd className="tabular text-right">{model.alias}</dd>
          <dt className="label-caps text-[var(--color-muted-text-on-dark)]">Version</dt>
          <dd className="tabular text-right">v{model.version}</dd>
          <dt className="label-caps text-[var(--color-muted-text-on-dark)]">Trained</dt>
          <dd className="tabular text-right">{model.trained_at.slice(0, 10)}</dd>
        </dl>
      ) : (
        <p className="mt-4 text-sm text-[var(--color-muted-text-on-dark)]">
          {health?.detail ?? "No model status available."}
        </p>
      )}
    </FadeIn>
  );
}
