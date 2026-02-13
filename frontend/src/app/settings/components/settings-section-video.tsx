import { Clapperboard } from "lucide-react";

interface SettingsSectionVideoProps {
  isSaving: boolean;
  transitionsEnabled: boolean;
  onToggleTransitions: () => void;
}

export function SettingsSectionVideo({ isSaving, transitionsEnabled, onToggleTransitions }: SettingsSectionVideoProps) {
  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-center gap-2">
          <Clapperboard className="w-4 h-4 text-black" />
          <h3 className="text-lg font-semibold text-black">Default Video Settings</h3>
        </div>
        <p className="mt-1 text-xs text-gray-500">Defaults that affect clip composition and rendering behavior.</p>
      </div>

      <div className="space-y-3 rounded-md border border-gray-200 bg-white p-3">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-medium text-black">Enable transitions</p>
            <p className="text-xs text-gray-500">Add transition effects between consecutive generated clips.</p>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={transitionsEnabled}
            onClick={onToggleTransitions}
            disabled={isSaving}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors disabled:opacity-50 ${
              transitionsEnabled ? "bg-blue-600" : "bg-gray-300"
            }`}
          >
            <span
              className={`inline-block h-5 w-5 transform rounded-full bg-white transition-transform ${
                transitionsEnabled ? "translate-x-5" : "translate-x-1"
              }`}
            />
          </button>
        </div>
      </div>
    </div>
  );
}
