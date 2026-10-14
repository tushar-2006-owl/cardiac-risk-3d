import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { Html, OrbitControls, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { ArteryMesh } from "./ArteryMesh";
import { TorsoMesh } from "./TorsoMesh";
import {
  CHAMBER_CONFIG,
  RegionId,
  VESSEL_CONFIG,
  VesselId,
  probabilityColor,
} from "../config/vessels";

const GLB_URL = "/models/heart.glb";

function CameraFocus({ target }: { target: RegionId | null }) {
  const { camera } = useThree();
  const controls = useThree((s) => s.controls) as unknown as { target: THREE.Vector3; update: () => void } | null;
  useEffect(() => {
    if (!target) return;
    const pos =
      target in VESSEL_CONFIG
        ? VESSEL_CONFIG[target as VesselId].path[2]
        : CHAMBER_CONFIG[target as keyof typeof CHAMBER_CONFIG].position;
    const dest = new THREE.Vector3(pos[0] * 2.2, pos[1] * 2.2, pos[2] * 2.2 + 2.4);
    const look = new THREE.Vector3(...pos);
    const start = camera.position.clone();
    let t = 0;
    const id = setInterval(() => {
      t += 0.08;
      camera.position.lerpVectors(start, dest, Math.min(1, t));
      if (controls?.target) controls.target.lerp(look, 0.12);
      if (t >= 1) clearInterval(id);
    }, 16);
    return () => clearInterval(id);
  }, [target, camera, controls]);
  return null;
}

function Chamber({
  id,
  selected,
  heartOpacity,
  onSelect,
  onHover,
}: {
  id: keyof typeof CHAMBER_CONFIG;
  selected: boolean;
  heartOpacity: number;
  onSelect: (id: RegionId) => void;
  onHover: (id: RegionId | null) => void;
}) {
  const cfg = CHAMBER_CONFIG[id];
  return (
    <mesh
      name={cfg.meshKey}
      position={cfg.position}
      scale={cfg.scale}
      onClick={(e) => {
        e.stopPropagation();
        onSelect(id);
      }}
      onPointerOver={(e) => {
        e.stopPropagation();
        onHover(id);
      }}
      onPointerOut={() => onHover(null)}
    >
      <sphereGeometry args={[0.7, 16, 12]} />
      <meshStandardMaterial
        color={selected ? "#fda4af" : "#fb7185"}
        transparent
        opacity={heartOpacity}
        roughness={0.55}
        metalness={0.05}
      />
    </mesh>
  );
}

function ProceduralHeart({
  selected,
  heartOpacity,
  showTorso,
  torsoOpacity,
  onSelect,
  onHover,
}: Pick<SceneProps, "selected" | "heartOpacity" | "showTorso" | "torsoOpacity" | "onSelect" | "onHover">) {
  return (
    <group>
      {showTorso && <TorsoMesh opacity={torsoOpacity} />}
      {(Object.keys(CHAMBER_CONFIG) as (keyof typeof CHAMBER_CONFIG)[]).map((id) => (
        <Chamber
          key={id}
          id={id}
          selected={selected === id}
          heartOpacity={heartOpacity}
          onSelect={onSelect}
          onHover={onHover}
        />
      ))}
    </group>
  );
}

function OptionalGltf({ url, fallback }: { url: string; fallback: React.ReactNode }) {
  const [ok, setOk] = useState<boolean | null>(null);
  useEffect(() => {
    fetch(url, { method: "HEAD" })
      .then((r) => setOk(r.ok))
      .catch(() => setOk(false));
  }, [url]);
  if (ok === null) return null;
  if (!ok) return <>{fallback}</>;
  return <GltfHeart url={url} />;
}

function GltfHeart({ url }: { url: string }) {
  const gltf = useGLTF(url);
  return <primitive object={gltf.scene} scale={1.4} />;
}

interface SceneProps {
  vessels: Record<VesselId, number>;
  selected: RegionId | null;
  hovered: RegionId | null;
  showLabels: boolean;
  heartOpacity: number;
  showTorso: boolean;
  torsoOpacity: number;
  autoRotate: boolean;
  onSelect: (id: RegionId) => void;
  onHover: (id: RegionId | null) => void;
}

function SceneBody(props: SceneProps) {
  const controls = useRef<any>(null);
  return (
    <>
      <ambientLight intensity={0.7} />
      <directionalLight position={[3, 4, 2]} intensity={0.9} />
      <directionalLight position={[-2, 1, -2]} intensity={0.25} />
      <Suspense fallback={null}>
        <OptionalGltf
          url={GLB_URL}
          fallback={
            <ProceduralHeart
              selected={props.selected}
              heartOpacity={props.heartOpacity}
              showTorso={props.showTorso}
              torsoOpacity={props.torsoOpacity}
              onSelect={props.onSelect}
              onHover={props.onHover}
            />
          }
        />
        {/* Arteries always overlay so model outputs drive named meshes */}
        {(Object.keys(VESSEL_CONFIG) as VesselId[]).map((id) => (
          <ArteryMesh
            key={`overlay-${id}`}
            id={id}
            probability={props.vessels[id]}
            selected={props.selected === id}
            hovered={props.hovered === id}
            showLabel={props.showLabels}
            onHover={props.onHover}
            onSelect={props.onSelect}
          />
        ))}
      </Suspense>
      <CameraFocus target={props.selected} />
      <OrbitControls
        ref={controls}
        makeDefault
        enableDamping
        dampingFactor={0.08}
        minDistance={1.6}
        maxDistance={8}
        autoRotate={props.autoRotate}
        autoRotateSpeed={0.8}
      />
    </>
  );
}

export function HeartScene(
  props: Omit<SceneProps, "hovered"> & {
    hovered: RegionId | null;
    onResetView?: (reset: () => void) => void;
  }
) {
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);

  return (
    <div className="relative h-full w-full">
      <Canvas
        dpr={[1, 1.5]}
        camera={{ position: [2.4, 0.4, 3.4], fov: 45 }}
        gl={{ antialias: true, powerPreference: "low-power" }}
        onPointerMissed={() => props.onSelect(props.selected as RegionId)}
      >
        <SceneBody
          {...props}
          onHover={(id) => {
            props.onHover(id);
          }}
        />
      </Canvas>
      {props.hovered && (
        <div className="pointer-events-none absolute left-3 top-3 rounded bg-slate-900/90 px-2 py-1 text-xs text-white">
          {props.hovered in VESSEL_CONFIG
            ? `${props.hovered}: ${((props.vessels as any)[props.hovered] * 100).toFixed(1)}% predicted risk`
            : `${CHAMBER_CONFIG[props.hovered as keyof typeof CHAMBER_CONFIG]?.label ?? props.hovered} — not predicted by the model`}
        </div>
      )}
      <Legend vessels={props.vessels} />
    </div>
  );
}

function Legend({ vessels }: { vessels: Record<VesselId, number> }) {
  return (
    <div className="absolute bottom-3 left-3 rounded-lg bg-slate-900/85 p-2 text-[11px] text-slate-100 shadow">
      <div className="mb-1 font-semibold">Predicted stenosis probability</div>
      <div
        className="mb-1 h-2 w-40 rounded"
        style={{
          background: "linear-gradient(90deg,#22c55e,#eab308,#f97316,#dc2626)",
        }}
      />
      <div className="mb-1 flex justify-between text-[10px] text-slate-300">
        <span>0</span>
        <span>0.33</span>
        <span>0.66</span>
        <span>1.0</span>
      </div>
      {(Object.keys(VESSEL_CONFIG) as VesselId[]).map((id) => (
        <div key={id} className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ background: probabilityColor(vessels[id]) }} />
          <span>
            {id}: {(vessels[id] * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}
