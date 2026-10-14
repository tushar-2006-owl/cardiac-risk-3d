import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { VESSEL_CONFIG, VesselId, probabilityColor } from "../config/vessels";

export function ArteryMesh({
  id,
  probability,
  selected,
  hovered,
  showLabel,
  onHover,
  onSelect,
}: {
  id: VesselId;
  probability: number;
  selected: boolean;
  hovered: boolean;
  showLabel: boolean;
  onHover: (id: VesselId | null) => void;
  onSelect: (id: VesselId) => void;
}) {
  const cfg = VESSEL_CONFIG[id];
  const mat = useRef<THREE.MeshStandardMaterial>(null);
  const target = useMemo(() => new THREE.Color(probabilityColor(probability)), [probability]);
  const current = useRef(new THREE.Color(probabilityColor(0.15)));

  const geometry = useMemo(() => {
    const curve = new THREE.CatmullRomCurve3(cfg.path.map((p) => new THREE.Vector3(...p)));
    return new THREE.TubeGeometry(curve, 48, selected || hovered ? 0.045 : 0.032, 8, false);
  }, [cfg.path, selected, hovered]);

  useFrame((_, dt) => {
    if (!mat.current) return;
    current.current.lerp(target, Math.min(1, dt * 4));
    mat.current.color.copy(current.current);
    mat.current.emissive.copy(current.current).multiplyScalar(selected || hovered ? 0.35 : 0.12);
  });

  const mid = cfg.path[Math.floor(cfg.path.length / 2)];

  return (
    <group>
      <mesh
        name={cfg.meshKey}
        geometry={geometry}
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover(id);
        }}
        onPointerOut={() => onHover(null)}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(id);
        }}
      >
        <meshStandardMaterial ref={mat} roughness={0.45} metalness={0.15} />
      </mesh>
      {showLabel && (
        <Html position={mid} distanceFactor={6} style={{ pointerEvents: "none" }}>
          <div className="rounded bg-slate-900/90 px-1.5 py-0.5 text-[10px] font-semibold text-white shadow">
            {id} {(probability * 100).toFixed(0)}%
          </div>
        </Html>
      )}
    </group>
  );
}
