import { DISCLAIMER } from "../config/vessels";

export function DisclaimerBanner({ compact = false }: { compact?: boolean }) {
  return (
    <div
      role="note"
      className={
        compact
          ? "bg-amber-950/80 px-3 py-1 text-center text-[11px] text-amber-100"
          : "border-b border-amber-700/50 bg-amber-900 px-4 py-2 text-center text-sm font-medium text-amber-50"
      }
    >
      {DISCLAIMER} Outputs are <em>predicted risk</em>, not a diagnosis. Tabular inputs do not localize lesions.
    </div>
  );
}
