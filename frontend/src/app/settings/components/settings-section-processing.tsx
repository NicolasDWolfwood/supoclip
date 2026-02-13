import { Bot, Cloud, Cpu, KeyRound, Settings2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { AiProvider, TranscriptionProvider } from "../settings-section-types";
import { AI_PROVIDERS, DEFAULT_AI_MODELS } from "../settings-section-types";

interface SettingsSectionProcessingProps {
  isSaving: boolean;
  transitionsEnabled: boolean;
  transcriptionProvider: TranscriptionProvider;
  aiProvider: AiProvider;
  aiModel: string;
  aiModelOptions: string[];
  hasLoadedAiModels: boolean;
  hasAiKeyForSelectedProvider: boolean;
  isLoadingAiModels: boolean;
  isSavingAssemblyKey: boolean;
  isSavingAiKey: boolean;
  assemblyApiKey: string;
  hasSavedAssemblyKey: boolean;
  hasAssemblyEnvFallback: boolean;
  assemblyKeyStatus: string | null;
  assemblyKeyError: string | null;
  aiApiKey: string;
  hasSavedAiKey: boolean;
  hasEnvAiFallback: boolean;
  aiKeyStatus: string | null;
  aiKeyError: string | null;
  aiModelStatus: string | null;
  aiModelError: string | null;
  onToggleTransitions: () => void;
  onTranscriptionProviderChange: (provider: TranscriptionProvider) => void;
  onAiProviderChange: (provider: AiProvider) => void;
  onAiModelChange: (model: string) => void;
  onAssemblyApiKeyChange: (value: string) => void;
  onSaveAssemblyKey: () => void;
  onDeleteAssemblyKey: () => void;
  onAiApiKeyChange: (value: string) => void;
  onSaveAiProviderKey: () => void;
  onDeleteAiProviderKey: () => void;
  onRefreshAiModels: () => void;
}

export function SettingsSectionProcessing({
  isSaving,
  transitionsEnabled,
  transcriptionProvider,
  aiProvider,
  aiModel,
  aiModelOptions,
  hasLoadedAiModels,
  hasAiKeyForSelectedProvider,
  isLoadingAiModels,
  isSavingAssemblyKey,
  isSavingAiKey,
  assemblyApiKey,
  hasSavedAssemblyKey,
  hasAssemblyEnvFallback,
  assemblyKeyStatus,
  assemblyKeyError,
  aiApiKey,
  hasSavedAiKey,
  hasEnvAiFallback,
  aiKeyStatus,
  aiKeyError,
  aiModelStatus,
  aiModelError,
  onToggleTransitions,
  onTranscriptionProviderChange,
  onAiProviderChange,
  onAiModelChange,
  onAssemblyApiKeyChange,
  onSaveAssemblyKey,
  onDeleteAssemblyKey,
  onAiApiKeyChange,
  onSaveAiProviderKey,
  onDeleteAiProviderKey,
  onRefreshAiModels,
}: SettingsSectionProcessingProps) {
  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-center gap-2">
          <Settings2 className="w-4 h-4 text-black" />
          <h3 className="text-lg font-semibold text-black">Default Processing Settings</h3>
        </div>
        <p className="mt-1 text-xs text-gray-500">API keys require explicit Save Key.</p>
      </div>

      <div className="space-y-3 rounded-md border border-gray-200 bg-white p-3">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-medium text-black">Enable transitions</p>
            <p className="text-xs text-gray-500">Add transition effects between consecutive clips.</p>
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
            <p className="text-sm font-medium text-black">Transcription Provider</p>
            <p className="text-xs text-gray-500">Choose local Whisper or AssemblyAI for transcript generation.</p>
          </div>
          <KeyRound className="w-4 h-4 text-gray-500 mt-0.5" />
        </div>

        <Select
          value={transcriptionProvider}
          onValueChange={(value) => onTranscriptionProviderChange(value as TranscriptionProvider)}
          disabled={isSaving || isSavingAssemblyKey}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select transcription provider" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="local">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4" />
                Local Whisper
              </div>
            </SelectItem>
            <SelectItem value="assemblyai">
              <div className="flex items-center gap-2">
                <Cloud className="w-4 h-4" />
                AssemblyAI
              </div>
            </SelectItem>
          </SelectContent>
        </Select>

        <p className="text-xs text-gray-500">
          {transcriptionProvider === "local"
            ? "Local mode uses the local worker queue and can run in parallel across workers."
            : "AssemblyAI mode uses a dedicated single-worker queue to avoid overloading remote transcription jobs."}
        </p>

        {transcriptionProvider === "assemblyai" && (
          <div className="space-y-2 rounded border border-gray-100 bg-gray-50 p-3">
            <label htmlFor="assembly-api-key" className="text-xs font-medium text-black">
              AssemblyAI API Key
            </label>
            <Input
              id="assembly-api-key"
              type="password"
              value={assemblyApiKey}
              onChange={(event) => onAssemblyApiKeyChange(event.target.value ?? "")}
              placeholder={
                hasSavedAssemblyKey ? "Saved key present (enter new key to replace)" : "Paste your AssemblyAI key"
              }
              disabled={isSaving || isSavingAssemblyKey}
            />
            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={isSaving || isSavingAssemblyKey || !assemblyApiKey.trim()}
                onClick={onSaveAssemblyKey}
              >
                {isSavingAssemblyKey ? "Saving..." : "Save Key"}
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                disabled={isSaving || isSavingAssemblyKey || !hasSavedAssemblyKey}
                onClick={onDeleteAssemblyKey}
              >
                Remove Saved Key
              </Button>
              <span className="text-xs text-gray-500">
                {hasSavedAssemblyKey
                  ? "Saved key available"
                  : hasAssemblyEnvFallback
                    ? "No saved key; using backend env fallback"
                    : "No key configured"}
              </span>
            </div>
            {assemblyKeyStatus && <p className="text-xs text-green-600">{assemblyKeyStatus}</p>}
            {assemblyKeyError && <p className="text-xs text-red-600">{assemblyKeyError}</p>}
          </div>
        )}
      </div>

      <div className="space-y-3 rounded-md border border-gray-200 bg-white p-3">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-medium text-black">AI Provider</p>
            <p className="text-xs text-gray-500">Choose which LLM provider analyzes transcripts to select clips.</p>
          </div>
          <Bot className="w-4 h-4 text-gray-500 mt-0.5" />
        </div>

        <Select
          value={aiProvider}
          onValueChange={(value) => onAiProviderChange(value as AiProvider)}
          disabled={isSaving || isSavingAiKey}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select AI provider" />
          </SelectTrigger>
          <SelectContent>
            {AI_PROVIDERS.map((provider) => (
              <SelectItem key={provider} value={provider}>
                {provider === "zai" ? "z.ai (GLM)" : provider.charAt(0).toUpperCase() + provider.slice(1)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <div className="space-y-2 rounded border border-gray-100 bg-gray-50 p-3">
          <label htmlFor="ai-provider-key" className="text-xs font-medium text-black">
            {aiProvider.toUpperCase()} API Key
          </label>
          <Input
            id="ai-provider-key"
            type="password"
            value={aiApiKey}
            onChange={(event) => onAiApiKeyChange(event.target.value ?? "")}
            placeholder={
              hasSavedAiKey
                ? `Saved ${aiProvider} key present (enter new key to replace)`
                : `Paste your ${aiProvider} API key`
            }
            disabled={isSaving || isSavingAiKey}
          />
          <div className="flex flex-wrap items-center gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={isSaving || isSavingAiKey || !aiApiKey.trim()}
              onClick={onSaveAiProviderKey}
            >
              {isSavingAiKey ? "Saving..." : "Save Key"}
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              disabled={isSaving || isSavingAiKey || !hasSavedAiKey}
              onClick={onDeleteAiProviderKey}
            >
              Remove Saved Key
            </Button>
            <span className="text-xs text-gray-500">
              {hasSavedAiKey
                ? "Saved key available"
                : hasEnvAiFallback
                  ? "No saved key; using backend env fallback"
                  : "No key configured"}
            </span>
          </div>
          {aiKeyStatus && <p className="text-xs text-green-600">{aiKeyStatus}</p>}
          {aiKeyError && <p className="text-xs text-red-600">{aiKeyError}</p>}
        </div>

        <div className="space-y-2 rounded border border-gray-100 bg-gray-50 p-3">
          <label htmlFor="ai-provider-model" className="text-xs font-medium text-black">
            {aiProvider.toUpperCase()} Model
          </label>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={isSaving || isLoadingAiModels || isSavingAiKey || !hasAiKeyForSelectedProvider}
              onClick={onRefreshAiModels}
            >
              {isLoadingAiModels ? "Loading Models..." : "Refresh Models"}
            </Button>
            {!hasAiKeyForSelectedProvider && (
              <span className="text-xs text-gray-500">Save a key (or configure env fallback) to load models.</span>
            )}
          </div>
          <Input
            id="ai-provider-model"
            list={`ai-model-options-${aiProvider}`}
            value={aiModel}
            onChange={(event) => onAiModelChange(event.target.value ?? "")}
            placeholder={`Default: ${DEFAULT_AI_MODELS[aiProvider]}`}
            disabled={isSaving}
          />
          <datalist id={`ai-model-options-${aiProvider}`}>
            {aiModelOptions.map((model) => (
              <option key={model} value={model} />
            ))}
          </datalist>
          <p className="text-xs text-gray-500">
            Clear the field to revert to the default model: {DEFAULT_AI_MODELS[aiProvider]}.
          </p>
          {hasLoadedAiModels && (
            <p className="text-xs text-gray-500">Loaded {aiModelOptions.length} model options directly from {aiProvider}.</p>
          )}
          {aiModelStatus && <p className="text-xs text-green-600">{aiModelStatus}</p>}
          {aiModelError && <p className="text-xs text-red-600">{aiModelError}</p>}
        </div>
      </div>
    </div>
  );
}
