import { ALL_REGIONS, RegionId } from "../config/vessels";

export function RegionList({
  selected,
  onSelect,
}: {
  selected: RegionId | null;
  onSelect: (id: RegionId) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {ALL_REGIONS.map((r) => (
        <button
          key={r.id}
          type="button"
          onClick={() => onSelect(r.id)}
          className={`rounded-full px-2 py-0.5 text-xs ${
            selected === r.id ? "bg-rose-500 text-white" : "bg-slate-800 text-slate-200 hover:bg-slate-700"
          }`}
        >
          {r.label}
          {r.kind === "chamber" ? " · n/a" : ""}
        </button>
      ))}
    </div>
  );
}
