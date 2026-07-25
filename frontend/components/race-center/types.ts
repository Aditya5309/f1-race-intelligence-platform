import type { DriverPrediction, ModelInfo, ProvenanceSchema } from "@/lib/api";

/**
 * Unified shape for both a historical prediction (GET /predictions/{id})
 * and an upcoming one (POST /predict) — mirrors
 * app/views/race_center.py's _normalize_upcoming_response() adapter, so
 * every rendering component below reads one shape and never forks on
 * `isUpcoming` itself.
 */
export interface RaceCenterPrediction {
  race_id: number;
  year: number;
  round: number;
  prediction_id: string;
  generated_at: string;
  model: ModelInfo;
  predictions: DriverPrediction[];
  actual_winner_driver_id: number | null;
  model_top1_hit: boolean | null;
  isUpcoming: boolean;
  materialization_status?: string;
  missing_inputs?: string[];
  caveats?: string[];
  provenance?: ProvenanceSchema;
}
