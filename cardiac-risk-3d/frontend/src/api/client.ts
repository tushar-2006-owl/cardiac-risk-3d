export interface FeatureSchema {
  name: string;
  display_label: string;
  group: string;
  type: "number" | "binary" | "categorical" | "ordinal";
  unit: string | null;
  min: number | null;
  max: number | null;
  default: number | string | null;
  categories: (string | number)[] | null;
  reference_range: { low: number; high: number } | null;
  notes?: string | null;
}

export interface ShapItem {
  feature: string;
  value: unknown;
  shap: number;
  direction: "increases risk" | "decreases risk";
  percent_contribution: number;
}

export interface PredictResponse {
  cad: { probability: number; label: "CAD" | "Normal"; risk_level: "low" | "medium" | "high" };
  vessels: Record<
    "LAD" | "LCX" | "RCA",
    { probability: number; risk_level: "low" | "medium" | "high"; mesh_key: string }
  >;
  explanations: Record<string, ShapItem[]>;
  disclaimer: string;
  global_shap?: Record<string, { feature: string; mean_abs_shap: number; percent: number }[]>;
  patient_values?: Record<string, unknown>;
}

export interface SamplePatient {
  id: string;
  label: string;
  risk_hint: string;
  demo_only: boolean;
  ground_truth?: Record<string, number>;
  features: Record<string, unknown>;
}

async function parse(res: Response) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  health: () => fetch("/health").then(parse),
  schema: () => fetch("/schema").then(parse) as Promise<{ features: FeatureSchema[]; disclaimer: string }>,
  metrics: () => fetch("/metrics").then(parse),
  samples: () => fetch("/patients/sample").then(parse) as Promise<{ patients: SamplePatient[] }>,
  predict: (features: Record<string, unknown>) =>
    fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(features),
    }).then(parse) as Promise<PredictResponse>,
};
