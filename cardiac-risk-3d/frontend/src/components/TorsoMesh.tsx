export function TorsoMesh({ opacity }: { opacity: number }) {
  return (
    <group name="torso">
      {/* Rib-cage-like ellipsoid shell */}
      <mesh rotation={[0.08, 0, 0]} position={[0, -0.1, 0]}>
        <sphereGeometry args={[1.55, 20, 14]} />
        <meshStandardMaterial
          color="#64748b"
          transparent
          opacity={opacity * 0.18}
          wireframe={false}
          side={2}
          depthWrite={false}
        />
      </mesh>
      <mesh rotation={[0.08, 0, 0]} position={[0, -0.1, 0]} scale={[1, 1.15, 0.72]}>
        <sphereGeometry args={[1.52, 12, 8]} />
        <meshStandardMaterial color="#94a3b8" transparent opacity={opacity * 0.35} wireframe />
      </mesh>
    </group>
  );
}
