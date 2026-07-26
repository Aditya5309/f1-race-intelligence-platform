/**
 * Thin typed fetch wrapper over the FastAPI backend (app/api.py). Every
 * route used here is read-only and unauthenticated (Decision 044). All
 * requests target the canonical /api/v1 prefix (Decision 045) — the
 * legacy unversioned alias exists only for the Streamlit client's
 * back-compat, this frontend has no reason to use it.
 *
 * NEXT_PUBLIC_API_URL must be set (frontend/.env.local, or the Vercel
 * project's env vars) — see app/config.py's Settings.cors_allow_origins
 * for the matching backend-side allow-list this origin needs to be added to.
 */

const API_BASE = `${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/api/v1`;

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new ApiError(res.status, detail?.detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new ApiError(res.status, detail?.detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

export interface ModelInfo {
  name: string;
  version: string;
  alias: string;
  run_id: string;
  trained_at: string;
  calibration: string;
  model_class: string;
}

export interface HealthResponse {
  status: string;
  api_version: string;
  model: ModelInfo | null;
  detail: string | null;
}

export interface RaceSummary {
  race_id: number;
  year: number;
  round: number;
  n_drivers: number;
}

export interface UpcomingRace {
  race_id: number;
  year: number;
  round: number;
  name: string;
  circuit_id: number;
  date: string;
}

export interface DriverPrediction {
  driver_id: number;
  driver_name: string | null;
  constructor_id: number | null;
  constructor_name: string | null;
  predicted_rank: number;
  win_probability: number;
  win_probability_raw: number;
}

export interface PredictionResponse {
  prediction_id: string;
  race_id: number;
  year: number;
  round: number;
  generated_at: string;
  model: ModelInfo;
  predictions: DriverPrediction[];
  actual_winner_driver_id: number | null;
  model_top1_hit: boolean | null;
}

export const getHealth = () => apiGet<HealthResponse>("/health");

export const getModel = () => apiGet<ModelInfo>("/model");

export const listRaces = (year?: number) =>
  apiGet<{ races: RaceSummary[] }>(year ? `/races?year=${year}` : "/races").then(
    (r) => r.races,
  );

/** null when no upcoming race is scheduled (a real, expected 404, not an error state). */
export async function getUpcomingRace(): Promise<UpcomingRace | null> {
  try {
    return await apiGet<UpcomingRace>("/races/upcoming");
  } catch (err) {
    if (err instanceof ApiError && (err.status === 404 || err.status === 503)) {
      return null;
    }
    throw err;
  }
}

export const getPredictions = (raceId: number) =>
  apiGet<PredictionResponse>(`/predictions/${raceId}`);

export interface ProvenanceSchema {
  model_version: string;
  model_alias: string;
  feature_schema_version: string;
  etl_snapshot_version: string;
  data_as_of: string;
  materialized_at: string;
  predicted_at: string;
  qualifying_status: string;
  completeness_status: string;
}

export interface UpcomingPredictionResponse {
  prediction_id: string;
  year: number;
  round: number;
  materialization_status: string; // "post_qualifying" | "pre_qualifying"
  missing_inputs: string[];
  generated_at: string;
  model: ModelInfo;
  predictions: DriverPrediction[];
  caveats: string[];
  provenance: ProvenanceSchema;
}

/**
 * POST /predict — scores the single next race with no result yet (Decision
 * 050 horizon=1; confirmed live 2026-07-24, not a stub — see
 * app/upcoming_prediction_service.py). 409 (race already has a result) and
 * 503 (materialization data unavailable) are real, expected outcomes here,
 * not just error-path edge cases — callers should handle both explicitly
 * rather than treating every non-2xx the same way.
 */
export const predictUpcoming = (year: number, round: number) =>
  apiPost<UpcomingPredictionResponse>("/predict", { year, round });

export interface SimulateGridResponse {
  race_id: number;
  driver_id: number;
  driver_name: string | null;
  field_size: number;
  real_grid_position: number | null;
  simulated_grid_position: number;
  pit_lane_start: boolean;
  real_win_probability: number;
  simulated_win_probability: number;
  field: DriverPrediction[];
  locked_qualifying_features: string[];
  locked_features: string[];
  model: ModelInfo;
}

export function simulateGrid(
  raceId: number,
  driverId: number,
  params: { gridPosition: number } | { pitLane: true },
) {
  const search =
    "pitLane" in params
      ? "pit_lane=true"
      : `grid_position=${params.gridPosition}`;
  return apiGet<SimulateGridResponse>(
    `/predictions/${raceId}/simulate/${driverId}?${search}`,
  );
}

export interface BaselineComparisonResponse {
  race_id: number;
  year: number;
  round: number;
  model: ModelInfo;
  baseline_name: string;
  baseline_description: string;
  model_predictions: DriverPrediction[];
  baseline_predictions: DriverPrediction[];
  actual_winner_driver_id: number | null;
  model_top1_hit: boolean | null;
  baseline_top1_hit: boolean | null;
}

export const getVsBaseline = (raceId: number) =>
  apiGet<BaselineComparisonResponse>(`/predictions/${raceId}/vs-baseline`);

export interface DriverStanding {
  position: number;
  driver_id: number;
  driver_name: string | null;
  points: number;
  wins: number;
}

export interface ConstructorStanding {
  position: number;
  constructor_id: number;
  constructor_name: string | null;
  points: number;
  wins: number;
}

export interface StandingsResponse {
  year: number;
  drivers: DriverStanding[];
  constructors: ConstructorStanding[];
}

/**
 * GET /standings/{year} — championship standings, display metadata only
 * (Decision 062, Frontend Overhaul Phase 3). Degrades to empty
 * drivers/constructors arrays (still a 200, never 404/503) when a season
 * has no recorded standings yet — render that as an empty-state, not an
 * error.
 */
export const getStandings = (year: number) =>
  apiGet<StandingsResponse>(`/standings/${year}`);

export interface ModelComparisonResponse {
  columns: string[];
  rows: Record<string, number | string | boolean | null>[];
}

/**
 * GET /model-comparison, /feature-importance, /shap-summary (Decision 063,
 * Frontend Overhaul Phase 4) — recorded analysis snapshots
 * (artifacts/analysis/, exported by scripts/export_analysis_data.py),
 * independent of model/features readiness. The remaining
 * reports/phase4_analysis/ artifacts (SHAP beeswarm/waterfall, calibration,
 * confusion matrix, ROC, precision-recall) have no route — see that
 * script's docstring for why.
 */
export const getModelComparison = () => apiGet<ModelComparisonResponse>("/model-comparison");

export interface FeatureImportanceRow {
  feature: string;
  importance: number;
  feature_class: string;
}

export interface ImportanceByClassRow {
  feature_class: string;
  total_importance: number;
}

export interface FeatureImportanceResponse {
  features: FeatureImportanceRow[];
  by_class: ImportanceByClassRow[];
}

export const getFeatureImportance = () =>
  apiGet<FeatureImportanceResponse>("/feature-importance");

export interface ShapSummaryRow {
  feature: string;
  mean_abs_shap: number;
  feature_class: string;
}

export interface ShapSummaryResponse {
  features: ShapSummaryRow[];
}

export const getShapSummary = () => apiGet<ShapSummaryResponse>("/shap-summary");

// ---------------------------------------------------------------------------
// Entity-stats route family (Decision 064, Frontend Overhaul Phase 5) —
// Driver Explorer / Compare Drivers / Team all share this. Every route
// degrades to zero/null/empty fields (never 404/503) when an entity has no
// data in the requested scope — not an error state to special-case.
// ---------------------------------------------------------------------------

export interface DriverStats {
  driver_id: number;
  year: number | null;
  races: number;
  wins: number;
  podiums: number;
  poles: number | null;
  points: number;
  avg_quali: number | null;
  avg_finish: number | null;
  avg_finish_classified: number | null;
  consistency_std: number | null;
}

export const getDriverStats = (driverId: number, year?: number) =>
  apiGet<DriverStats>(`/drivers/${driverId}/stats${year ? `?year=${year}` : ""}`);

export interface RaceResultRow {
  race_id: number;
  year: number;
  round: number;
  quali_position: number | null;
  finish: number | null;
}

export const getDriverRaceResults = (driverId: number, year?: number) =>
  apiGet<{ driver_id: number; year: number | null; results: RaceResultRow[] }>(
    `/drivers/${driverId}/race-results${year ? `?year=${year}` : ""}`,
  ).then((r) => r.results);

export interface StandingsProgressionRow {
  round: number;
  points: number;
  position: number;
  wins: number;
}

export const getDriverStandingsProgression = (driverId: number, year: number) =>
  apiGet<{ driver_id: number; year: number; progression: StandingsProgressionRow[] }>(
    `/drivers/${driverId}/standings-progression?year=${year}`,
  ).then((r) => r.progression);

export const getDriverRadar = (driverId: number, year: number) =>
  apiGet<{ driver_id: number; year: number; scores: Record<string, number> | null }>(
    `/drivers/${driverId}/radar?year=${year}`,
  ).then((r) => r.scores);

export interface ConstructorCatalogEntry {
  constructor_id: number;
  name: string;
  nationality: string | null;
}

export const getConstructors = () =>
  apiGet<{ constructors: ConstructorCatalogEntry[] }>("/constructors").then(
    (r) => r.constructors,
  );

export interface ConstructorStats {
  constructor_id: number;
  year: number | null;
  races: number;
  wins: number;
  podiums: number;
  points: number;
  avg_quali: number | null;
}

export const getConstructorStats = (constructorId: number, year?: number) =>
  apiGet<ConstructorStats>(`/constructors/${constructorId}/stats${year ? `?year=${year}` : ""}`);

export const getConstructorRaceResults = (constructorId: number, year?: number) =>
  apiGet<{ constructor_id: number; year: number | null; results: RaceResultRow[] }>(
    `/constructors/${constructorId}/race-results${year ? `?year=${year}` : ""}`,
  ).then((r) => r.results);

export const getConstructorDrivers = (constructorId: number, year: number) =>
  apiGet<{ constructor_id: number; year: number; drivers: string[] }>(
    `/constructors/${constructorId}/drivers?year=${year}`,
  ).then((r) => r.drivers);

// ---------------------------------------------------------------------------
// Circuit-stats routes (Decision 065, Frontend Overhaul Phase 6) — Circuit
// Explorer's data need, standalone from the entity-stats family above but
// following the same conventions (independent of model/features readiness,
// degrades to zero/null/empty rather than 404).
// ---------------------------------------------------------------------------

export interface CircuitCatalogEntry {
  circuit_id: number;
  name: string;
  country: string | null;
  location: string | null;
  races_held: number;
}

export const getCircuits = () =>
  apiGet<{ circuits: CircuitCatalogEntry[] }>("/circuits").then((r) => r.circuits);

export interface CircuitLayout {
  circuit_id: number;
  coordinates: [number, number][] | null;
  attribution: string | null;
}

export const getCircuitLayout = (circuitId: number) =>
  apiGet<CircuitLayout>(`/circuits/${circuitId}/layout`);

export interface FastestLapInfo {
  time: string;
  year: number;
  driver: string;
}

export interface CircuitWinnerRow {
  year: number;
  driver: string;
  constructor: string | null;
}

export interface MostWinsRow {
  driver: string;
  wins: number;
}

export interface CircuitStats {
  circuit_id: number;
  races_held: number;
  first_year: number | null;
  last_year: number | null;
  dnf_rate: number | null;
  avg_position_change: number | null;
  fastest_lap: FastestLapInfo | null;
  most_wins: MostWinsRow[];
  winners_by_year: CircuitWinnerRow[];
}

export const getCircuitStats = (circuitId: number) =>
  apiGet<CircuitStats>(`/circuits/${circuitId}/stats`);
