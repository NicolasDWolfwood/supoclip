interface SettingsSectionVideoProps {
  isSaving: boolean;
  transitionsEnabled: boolean;
  reviewBeforeRenderEnabled: boolean;
  timelineEditorEnabled: boolean;
  onToggleTransitions: () => void;
  onToggleReviewBeforeRender: () => void;
  onToggleTimelineEditor: () => void;
}

export function SettingsSectionVideo({
  isSaving,
  transitionsEnabled,
  reviewBeforeRenderEnabled,
  timelineEditorEnabled,
  onToggleTransitions,
  onToggleReviewBeforeRender,
  onToggleTimelineEditor,
}: SettingsSectionVideoProps) {
  return (
    <div className="space-y-4">
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
      <div className="space-y-3 rounded-md border border-gray-200 bg-white p-3">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-medium text-black">Review clips before rendering</p>
            <p className="text-xs text-gray-500">
              Pause at review stage by default for new tasks so timing/text can be edited before final render.
            </p>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={reviewBeforeRenderEnabled}
            onClick={onToggleReviewBeforeRender}
            disabled={isSaving}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors disabled:opacity-50 ${
              reviewBeforeRenderEnabled ? "bg-blue-600" : "bg-gray-300"
            }`}
          >
            <span
              className={`inline-block h-5 w-5 transform rounded-full bg-white transition-transform ${
                reviewBeforeRenderEnabled ? "translate-x-5" : "translate-x-1"
              }`}
            />
          </button>
        </div>
      </div>
      <div className="space-y-3 rounded-md border border-gray-200 bg-white p-3">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-medium text-black">Enable timeline editor in review</p>
            <p className="text-xs text-gray-500">
              Show the source video timeline with draggable clip ranges by default for new tasks.
            </p>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={timelineEditorEnabled}
            onClick={onToggleTimelineEditor}
            disabled={isSaving}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors disabled:opacity-50 ${
              timelineEditorEnabled ? "bg-blue-600" : "bg-gray-300"
            }`}
          >
            <span
              className={`inline-block h-5 w-5 transform rounded-full bg-white transition-transform ${
                timelineEditorEnabled ? "translate-x-5" : "translate-x-1"
              }`}
            />
          </button>
        </div>
      </div>
    </div>
  );
}
