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
