import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { AiProvider, ZaiRoutingMode } from "../settings-section-types";
import { AI_PROVIDERS, DEFAULT_AI_MODELS } from "../settings-section-types";

interface SettingsSectionAiProps {
  isSaving: boolean;
  aiProvider: AiProvider;
  aiModel: string;
  aiModelOptions: string[];
  hasLoadedAiModels: boolean;
  hasAiKeyForSelectedProvider: boolean;
  isLoadingAiModels: boolean;
  isSavingAiKey: boolean;
  aiApiKey: string;
  hasSavedAiKey: boolean;
  hasEnvAiFallback: boolean;
  ollamaServerUrl: string;
  hasSavedOllamaServer: boolean;
  hasEnvOllamaServer: boolean;
  aiKeyStatus: string | null;
  aiKeyError: string | null;
  aiModelStatus: string | null;
  aiModelError: string | null;
  selectedZaiKeyProfile: "subscription" | "metered";
  zaiRoutingMode: ZaiRoutingMode;
  zaiProfileApiKey: string;
  hasSavedZaiSubscriptionKey: boolean;
  hasSavedZaiMeteredKey: boolean;
  onAiProviderChange: (provider: AiProvider) => void;
  onAiModelChange: (model: string) => void;
  onAiApiKeyChange: (value: string) => void;
  onSaveAiProviderKey: () => void;
  onDeleteAiProviderKey: () => void;
  onOllamaServerUrlChange: (value: string) => void;
  onSaveOllamaServer: () => void;
  onDeleteOllamaServer: () => void;
  onRefreshAiModels: () => void;
  onSelectedZaiKeyProfileChange: (profile: "subscription" | "metered") => void;
  onZaiRoutingModeChange: (mode: ZaiRoutingMode) => void;
  onZaiProfileApiKeyChange: (value: string) => void;
  onSaveZaiProfileKey: () => void;
  onDeleteZaiProfileKey: () => void;
}

export function SettingsSectionAi({
  isSaving,
  aiProvider,
  aiModel,
  aiModelOptions,
  hasLoadedAiModels,
  hasAiKeyForSelectedProvider,
  isLoadingAiModels,
  isSavingAiKey,
  aiApiKey,
  hasSavedAiKey,
  hasEnvAiFallback,
  ollamaServerUrl,
  hasSavedOllamaServer,
  hasEnvOllamaServer,
  aiKeyStatus,
  aiKeyError,
  aiModelStatus,
  aiModelError,
  selectedZaiKeyProfile,
  zaiRoutingMode,
  zaiProfileApiKey,
  hasSavedZaiSubscriptionKey,
  hasSavedZaiMeteredKey,
  onAiProviderChange,
  onAiModelChange,
  onAiApiKeyChange,
  onSaveAiProviderKey,
  onDeleteAiProviderKey,
  onOllamaServerUrlChange,
  onSaveOllamaServer,
  onDeleteOllamaServer,
  onRefreshAiModels,
  onSelectedZaiKeyProfileChange,
  onZaiRoutingModeChange,
  onZaiProfileApiKeyChange,
  onSaveZaiProfileKey,
  onDeleteZaiProfileKey,
}: SettingsSectionAiProps) {
  return (
    <div className="space-y-4">
      <div className="space-y-3">
        <div>
          <p className="text-sm font-medium text-black">Provider</p>
          <p className="text-xs text-gray-500">Choose which LLM provider analyzes transcripts to select clips.</p>
        </div>

        <Select
          value={aiProvider}
          onValueChange={(value) => onAiProviderChange(value as AiProvider)}
          disabled={isSaving || isSavingAiKey}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select provider" />
          </SelectTrigger>
          <SelectContent>
            {AI_PROVIDERS.map((provider) => (
              <SelectItem key={provider} value={provider}>
                {provider === "zai"
                  ? "z.ai (GLM)"
                  : provider === "ollama"
                    ? "Ollama"
                    : provider.charAt(0).toUpperCase() + provider.slice(1)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {aiProvider === "zai" ? (
          <div className="space-y-2 rounded border border-gray-100 bg-gray-50 p-3">
            <label className="text-xs font-medium text-black">z.ai Key Routing</label>
            <Select
              value={zaiRoutingMode}
              onValueChange={(value) => onZaiRoutingModeChange(value as ZaiRoutingMode)}
              disabled={isSaving || isSavingAiKey}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select routing mode" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Auto (subscription then metered)</SelectItem>
                <SelectItem value="subscription">Subscription only</SelectItem>
                <SelectItem value="metered">Metered only</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-gray-500">
              Auto retries with your metered key if the subscription key is out of balance/package.
            </p>
            <div className="grid gap-2 sm:grid-cols-2">
              <div className="space-y-1">
                <label className="text-xs font-medium text-black">Profile to edit</label>
                <Select
                  value={selectedZaiKeyProfile}
                  onValueChange={(value) => onSelectedZaiKeyProfileChange(value as "subscription" | "metered")}
                  disabled={isSaving || isSavingAiKey}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select profile" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="subscription">Subscription</SelectItem>
                    <SelectItem value="metered">Metered</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium text-black">Saved profile keys</label>
                <p className="text-xs text-gray-600">
                  Subscription: {hasSavedZaiSubscriptionKey ? "yes" : "no"} | Metered: {hasSavedZaiMeteredKey ? "yes" : "no"}
                </p>
                {!hasSavedZaiSubscriptionKey && !hasSavedZaiMeteredKey && hasEnvAiFallback && (
                  <p className="text-xs text-gray-500">No z.ai profile key saved; backend env fallback is available.</p>
                )}
              </div>
            </div>
            <label htmlFor="zai-profile-key" className="text-xs font-medium text-black">
              z.ai {selectedZaiKeyProfile} API Key
            </label>
            <Input
              id="zai-profile-key"
              type="password"
              value={zaiProfileApiKey}
              onChange={(event) => onZaiProfileApiKeyChange(event.target.value ?? "")}
              placeholder={`Paste your z.ai ${selectedZaiKeyProfile} key`}
              disabled={isSaving || isSavingAiKey}
            />
            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={isSaving || isSavingAiKey || !zaiProfileApiKey.trim()}
                onClick={onSaveZaiProfileKey}
              >
                {isSavingAiKey ? "Saving..." : "Save Profile Key"}
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                disabled={
                  isSaving ||
                  isSavingAiKey ||
                  !(selectedZaiKeyProfile === "subscription" ? hasSavedZaiSubscriptionKey : hasSavedZaiMeteredKey)
                }
                onClick={onDeleteZaiProfileKey}
              >
                Remove Profile Key
              </Button>
            </div>
            {aiKeyStatus && <p className="text-xs text-green-600">{aiKeyStatus}</p>}
            {aiKeyError && <p className="text-xs text-red-600">{aiKeyError}</p>}
          </div>
        ) : aiProvider === "ollama" ? (
          <div className="space-y-2 rounded border border-gray-100 bg-gray-50 p-3">
            <label htmlFor="ollama-server-url" className="text-xs font-medium text-black">
              Ollama Server URL
            </label>
            <Input
              id="ollama-server-url"
              type="text"
              value={ollamaServerUrl}
              onChange={(event) => onOllamaServerUrlChange(event.target.value ?? "")}
              placeholder="http://localhost:11434"
              disabled={isSaving || isSavingAiKey}
            />
            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={isSaving || isSavingAiKey || !ollamaServerUrl.trim()}
                onClick={onSaveOllamaServer}
              >
                {isSavingAiKey ? "Saving..." : "Save Server"}
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                disabled={isSaving || isSavingAiKey || !hasSavedOllamaServer}
                onClick={onDeleteOllamaServer}
              >
                Remove Saved Server
              </Button>
              <span className="text-xs text-gray-500">
                {hasSavedOllamaServer
                  ? "Saved server URL available"
                  : hasEnvOllamaServer
                    ? "No saved server; using backend env fallback"
                    : ollamaServerUrl.trim()
                      ? "Using default backend Ollama URL"
                      : "No server configured"}
              </span>
            </div>
            {aiKeyStatus && <p className="text-xs text-green-600">{aiKeyStatus}</p>}
            {aiKeyError && <p className="text-xs text-red-600">{aiKeyError}</p>}
          </div>
        ) : (
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
        )}

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
              <span className="text-xs text-gray-500">
                {aiProvider === "ollama"
                  ? "Save an Ollama server URL (or configure OLLAMA_BASE_URL) to load models."
                  : "Save a key (or configure env fallback) to load models."}
              </span>
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
