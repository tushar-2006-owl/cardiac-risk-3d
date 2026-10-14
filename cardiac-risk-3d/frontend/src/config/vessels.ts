export type RiskLevel = "low" | "medium" | "high";

export type VesselId = "LAD" | "LCX" | "RCA";

export type RegionId =
  | VesselId
  | "LV"
  | "RV"
  | "LA"
  | "RA";

export interface VesselConfig {
  id: VesselId;
  meshKey: VesselId;
  label: string;
  fullName: string;
  description: string;
  supplies: RegionId[];
  /** Catmull-Rom points in local heart space. */
  path: [number, number, number][];
}

export interface ChamberConfig {
  id: Exclude<RegionId, VesselId>;
  meshKey: string;
  label: string;
  description: string;
  suppliedBy: VesselId[];
  position: [number, number, number];
  scale: [number, number, number];
}

/** Single source of truth: model outputs → named artery meshes. */
export const VESSEL_CONFIG: Record<VesselId, VesselConfig> = {
  LAD: {
    id: "LAD",
    meshKey: "LAD",
    label: "LAD",
    fullName: "Left anterior descending artery",
    description:
      "Runs in the anterior interventricular groove toward the apex. Predicted stenosis probability colors this mesh. Tabular inputs do not localize a lesion along the vessel.",
    supplies: ["LV", "RV"],
    path: [
      [0.12, 0.55, 0.22],
      [0.08, 0.25, 0.38],
      [0.02, -0.15, 0.42],
      [0.0, -0.55, 0.28],
      [0.02, -0.85, 0.05],
    ],
  },
  LCX: {
    id: "LCX",
    meshKey: "LCX",
    label: "LCX",
    fullName: "Left circumflex artery",
    description:
      "Wraps leftward in the atrioventricular groove toward the posterior-left heart. Color encodes predicted stenosis probability only.",
    supplies: ["LV", "LA"],
    path: [
      [0.12, 0.55, 0.18],
      [0.42, 0.42, 0.05],
      [0.55, 0.18, -0.22],
      [0.4, -0.05, -0.42],
      [0.12, -0.12, -0.5],
    ],
  },
  RCA: {
    id: "RCA",
    meshKey: "RCA",
    label: "RCA",
    fullName: "Right coronary artery",
    description:
      "Follows the right atrioventricular groove. Color encodes predicted stenosis probability only.",
    supplies: ["RV", "RA"],
    path: [
      [-0.08, 0.5, 0.2],
      [-0.4, 0.35, 0.12],
      [-0.52, 0.08, -0.08],
      [-0.38, -0.18, -0.32],
      [-0.08, -0.22, -0.45],
    ],
  },
};

export const CHAMBER_CONFIG: Record<Exclude<RegionId, VesselId>, ChamberConfig> = {
  LV: {
    id: "LV",
    meshKey: "LV",
    label: "Left ventricle",
    description: "Main pumping chamber. Typically supplied by LAD and LCX. Not predicted by the model.",
    suppliedBy: ["LAD", "LCX"],
    position: [0.18, -0.15, 0.05],
    scale: [0.55, 0.72, 0.5],
  },
  RV: {
    id: "RV",
    meshKey: "RV",
    label: "Right ventricle",
    description: "Anterior-right chamber. Typically supplied by RCA (and LAD septals). Not predicted by the model.",
    suppliedBy: ["RCA", "LAD"],
    position: [-0.22, -0.12, 0.18],
    scale: [0.48, 0.62, 0.42],
  },
  LA: {
    id: "LA",
    meshKey: "LA",
    label: "Left atrium",
    description: "Posterior-superior chamber. Circumflex branches often contribute. Not predicted by the model.",
    suppliedBy: ["LCX"],
    position: [0.16, 0.42, -0.22],
    scale: [0.38, 0.28, 0.32],
  },
  RA: {
    id: "RA",
    meshKey: "RA",
    label: "Right atrium",
    description: "Right-superior chamber, typically RCA territory. Not predicted by the model.",
    suppliedBy: ["RCA"],
    position: [-0.22, 0.4, -0.08],
    scale: [0.34, 0.28, 0.3],
  },
};

export const ALL_REGIONS: { id: RegionId; label: string; kind: "vessel" | "chamber" }[] = [
  { id: "LAD", label: "LAD", kind: "vessel" },
  { id: "LCX", label: "LCX", kind: "vessel" },
  { id: "RCA", label: "RCA", kind: "vessel" },
  { id: "LV", label: "Left ventricle", kind: "chamber" },
  { id: "RV", label: "Right ventricle", kind: "chamber" },
  { id: "LA", label: "Left atrium", kind: "chamber" },
  { id: "RA", label: "Right atrium", kind: "chamber" },
];

export const DISCLAIMER =
  "For decision support and educational purposes only. Not a substitute for formal diagnostic imaging or clinical judgment.";

export function probabilityColor(p: number): string {
  const t = Math.min(1, Math.max(0, p));
  // green → yellow → orange → red
  const stops: [number, [number, number, number]][] = [
    [0, [34, 197, 94]],
    [0.33, [234, 179, 8]],
    [0.66, [249, 115, 22]],
    [1, [220, 38, 38]],
  ];
  let a = stops[0];
  let b = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i][0] && t <= stops[i + 1][0]) {
      a = stops[i];
      b = stops[i + 1];
      break;
    }
  }
  const u = (t - a[0]) / (b[0] - a[0] || 1);
  const rgb = a[1].map((c, i) => Math.round(c + (b[1][i] - c) * u));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}
