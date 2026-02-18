import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type {
  AiProvider,
  OllamaAuthMode,
  OllamaProfileSummary,
  ZaiRoutingMode,
} from "../settings-section-types";
import { AI_PROVIDERS, DEFAULT_AI_MODELS, OLLAMA_AUTH_MODES } from "../settings-section-types";

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
  ollamaProfiles: OllamaProfileSummary[];
  selectedOllamaProfile: string;
  newOllamaProfileName: string;
  ollamaAuthMode: OllamaAuthMode;
  ollamaAuthHeaderName: string;
  ollamaAuthToken: string;
  ollamaTimeoutSeconds: number;
  ollamaMaxRetries: number;
  ollamaRetryBackoffMs: number;
  isTestingOllamaConnection: boolean;
  ollamaConnectionStatus: string | null;
  ollamaConnectionError: string | null;
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
  onSelectedOllamaProfileChange: (value: string) => void;
  onNewOllamaProfileNameChange: (value: string) => void;
  onCreateOllamaProfile: () => void;
  onSaveOllamaProfile: () => void;
  onDeleteOllamaProfile: () => void;
  onSetDefaultOllamaProfile: () => void;
  onOllamaAuthModeChange: (value: OllamaAuthMode) => void;
  onOllamaAuthHeaderNameChange: (value: string) => void;
  onOllamaAuthTokenChange: (value: string) => void;
  onOllamaTimeoutSecondsChange: (value: number) => void;
  onOllamaMaxRetriesChange: (value: number) => void;
  onOllamaRetryBackoffMsChange: (value: number) => void;
  onSaveOllamaRequestControls: () => void;
  onTestOllamaConnection: () => void;
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
  ollamaProfiles,
  selectedOllamaProfile,
  newOllamaProfileName,
  ollamaAuthMode,
  ollamaAuthHeaderName,
  ollamaAuthToken,
  ollamaTimeoutSeconds,
  ollamaMaxRetries,
  ollamaRetryBackoffMs,
  isTestingOllamaConnection,
  ollamaConnectionStatus,
  ollamaConnectionError,
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
  onSelectedOllamaProfileChange,
  onNewOllamaProfileNameChange,
  onCreateOllamaProfile,
  onSaveOllamaProfile,
  onDeleteOllamaProfile,
  onSetDefaultOllamaProfile,
  onOllamaAuthModeChange,
  onOllamaAuthHeaderNameChange,
  onOllamaAuthTokenChange,
  onOllamaTimeoutSecondsChange,
  onOllamaMaxRetriesChange,
  onOllamaRetryBackoffMsChange,
  onSaveOllamaRequestControls,
  onTestOllamaConnection,
  onRefreshAiModels,
  onSelectedZaiKeyProfileChange,
  onZaiRoutingModeChange,
  onZaiProfileApiKeyChange,
  onSaveZaiProfileKey,
  onDeleteZaiProfileKey,
}: SettingsSectionAiProps) {
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
  const [shouldFilterModelOptions, setShouldFilterModelOptions] = useState(false);
  const modelMenuRef = useRef<HTMLDivElement | null>(null);
  const selectedProfileMeta = ollamaProfiles.find((profile) => profile.profile_name === selectedOllamaProfile);
  const normalizedModelQuery = aiModel.trim().toLowerCase();

  const visibleAiModelOptions = useMemo(() => {
    if (!isModelMenuOpen) {
      return [];
    }
    if (!shouldFilterModelOptions || normalizedModelQuery.length === 0) {
      return aiModelOptions;
    }
    return aiModelOptions.filter((model) => model.toLowerCase().includes(normalizedModelQuery));
  }, [aiModelOptions, isModelMenuOpen, normalizedModelQuery, shouldFilterModelOptions]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (!modelMenuRef.current) {
        return;
      }
      if (!modelMenuRef.current.contains(event.target as Node)) {
        setIsModelMenuOpen(false);
        setShouldFilterModelOptions(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const openModelMenu = () => {
    setIsModelMenuOpen(true);
    setShouldFilterModelOptions(false);
  };

  const toggleModelMenu = () => {
    setIsModelMenuOpen((prev) => {
      const next = !prev;
      if (next) {
        setShouldFilterModelOptions(false);
      } else {
        setShouldFilterModelOptions(false);
      }
      return next;
    });
  };

  const closeModelMenu = () => {
    setIsModelMenuOpen(false);
    setShouldFilterModelOptions(false);
  };

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
          <div className="space-y-3 rounded border border-gray-100 bg-gray-50 p-3">
            <div className="grid gap-2 sm:grid-cols-2">
              <div className="space-y-1">
                <label className="text-xs font-medium text-black">Profile</label>
                <Select
                  value={selectedOllamaProfile || "__none__"}
                  onValueChange={(value) => onSelectedOllamaProfileChange(value === "__none__" ? "" : value)}
                  disabled={isSaving || isSavingAiKey}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select profile" />
                  </SelectTrigger>
                  <SelectContent>
                    {ollamaProfiles.length === 0 && <SelectItem value="__none__">No saved profiles</SelectItem>}
                    {ollamaProfiles.map((profile) => (
                      <SelectItem key={profile.profile_name} value={profile.profile_name}>
                        {profile.profile_name}
                        {profile.is_default ? " (default)" : ""}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex flex-wrap items-end gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={isSaving || isSavingAiKey || !selectedOllamaProfile || selectedProfileMeta?.is_default}
                  onClick={onSetDefaultOllamaProfile}
                >
                  Set Default
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  disabled={isSaving || isSavingAiKey || !selectedOllamaProfile}
                  onClick={onDeleteOllamaProfile}
                >
                  Delete Profile
                </Button>
              </div>
            </div>

            <div className="grid gap-2 sm:grid-cols-[1fr_auto]">
              <Input
                value={newOllamaProfileName}
                onChange={(event) => onNewOllamaProfileNameChange(event.target.value ?? "")}
                placeholder="new profile name"
                disabled={isSaving || isSavingAiKey}
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={isSaving || isSavingAiKey || !newOllamaProfileName.trim()}
                onClick={onCreateOllamaProfile}
              >
                Create Profile
              </Button>
            </div>

            <div className="space-y-1">
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
            </div>

            <div className="grid gap-2 sm:grid-cols-2">
              <div className="space-y-1">
                <label className="text-xs font-medium text-black">Auth Mode</label>
                <Select
                  value={ollamaAuthMode}
                  onValueChange={(value) => onOllamaAuthModeChange(value as OllamaAuthMode)}
                  disabled={isSaving || isSavingAiKey}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Auth mode" />
                  </SelectTrigger>
                  <SelectContent>
                    {OLLAMA_AUTH_MODES.map((mode) => (
                      <SelectItem key={mode} value={mode}>
                        {mode}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {ollamaAuthMode === "custom_header" && (
                <div className="space-y-1">
                  <label className="text-xs font-medium text-black">Header Name</label>
                  <Input
                    value={ollamaAuthHeaderName}
                    onChange={(event) => onOllamaAuthHeaderNameChange(event.target.value ?? "")}
                    placeholder="X-API-Key"
                    disabled={isSaving || isSavingAiKey}
                  />
                </div>
              )}
            </div>

            {ollamaAuthMode !== "none" && (
              <div className="space-y-1">
                <label className="text-xs font-medium text-black">Auth Token (write-only)</label>
                <Input
                  type="password"
                  value={ollamaAuthToken}
                  onChange={(event) => onOllamaAuthTokenChange(event.target.value ?? "")}
                  placeholder="Leave blank to keep existing token"
                  disabled={isSaving || isSavingAiKey}
                />
              </div>
            )}

            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={isSaving || isSavingAiKey || !ollamaServerUrl.trim()}
                onClick={onSaveOllamaProfile}
              >
                {isSavingAiKey ? "Saving..." : "Save Profile"}
              </Button>
              <span className="text-xs text-gray-500">
                {hasSavedOllamaServer
                  ? `Saved profiles: ${ollamaProfiles.length}`
                  : hasEnvOllamaServer
                    ? "No saved profile; using backend env fallback"
                    : "No saved profile configured"}
              </span>
            </div>

            <div className="grid gap-2 sm:grid-cols-3">
              <div className="space-y-1">
                <label className="text-xs font-medium text-black">Timeout (s)</label>
                <Input
                  type="number"
                  min={1}
                  max={600}
                  value={String(ollamaTimeoutSeconds)}
                  onChange={(event) => onOllamaTimeoutSecondsChange(Number(event.target.value))}
                  disabled={isSaving || isSavingAiKey}
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium text-black">Max Retries</label>
                <Input
                  type="number"
                  min={0}
                  max={10}
                  value={String(ollamaMaxRetries)}
                  onChange={(event) => onOllamaMaxRetriesChange(Number(event.target.value))}
                  disabled={isSaving || isSavingAiKey}
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium text-black">Backoff (ms)</label>
                <Input
                  type="number"
                  min={0}
                  max={30000}
                  value={String(ollamaRetryBackoffMs)}
                  onChange={(event) => onOllamaRetryBackoffMsChange(Number(event.target.value))}
                  disabled={isSaving || isSavingAiKey}
                />
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Button type="button" variant="outline" size="sm" disabled={isSaving || isSavingAiKey} onClick={onSaveOllamaRequestControls}>
                Save Request Controls
              </Button>
              <Button
                type="button"
                variant="secondary"
                size="sm"
                disabled={isSaving || isSavingAiKey || isTestingOllamaConnection}
                onClick={onTestOllamaConnection}
              >
                {isTestingOllamaConnection ? "Testing..." : "Test Connection"}
              </Button>
            </div>

            {ollamaConnectionStatus && <p className="text-xs text-green-600">{ollamaConnectionStatus}</p>}
            {ollamaConnectionError && <p className="text-xs text-red-600">{ollamaConnectionError}</p>}
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
                  ? "Configure an Ollama profile/server to load models."
                  : "Save a key (or configure env fallback) to load models."}
              </span>
            )}
          </div>
          <div
            ref={modelMenuRef}
            className="relative"
            onBlurCapture={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
                closeModelMenu();
              }
            }}
          >
            <div className="flex items-center gap-2">
              <Input
                id="ai-provider-model"
                value={aiModel}
                onFocus={() => {
                  if (!isSaving) {
                    openModelMenu();
                  }
                }}
                onChange={(event) => {
                  onAiModelChange(event.target.value ?? "");
                  if (!isSaving) {
                    setIsModelMenuOpen(true);
                    setShouldFilterModelOptions(true);
                  }
                }}
                onKeyDown={(event) => {
                  if (event.key === "ArrowDown" && !isModelMenuOpen) {
                    event.preventDefault();
                    openModelMenu();
                  }
                  if (event.key === "Escape" || event.key === "Tab") {
                    closeModelMenu();
                  }
                }}
                placeholder={`Default: ${DEFAULT_AI_MODELS[aiProvider]}`}
                disabled={isSaving}
              />
              <Button
                type="button"
                variant="outline"
                size="icon"
                aria-label="Show model options"
                disabled={isSaving}
                onClick={toggleModelMenu}
              >
                <ChevronDown className="h-4 w-4" />
              </Button>
            </div>
            {isModelMenuOpen && (
              <div className="absolute z-20 mt-1 max-h-56 w-full overflow-y-auto rounded-md border border-gray-200 bg-white shadow-lg">
                {visibleAiModelOptions.length > 0 ? (
                  visibleAiModelOptions.map((model) => (
                    <button
                      key={model}
                      type="button"
                      className="block w-full px-3 py-2 text-left text-sm text-black hover:bg-gray-50"
                      onMouseDown={(event) => {
                        event.preventDefault();
                      }}
                      onClick={() => {
                        onAiModelChange(model);
                        closeModelMenu();
                      }}
                    >
                      {model}
                    </button>
                  ))
                ) : (
                  <p className="px-3 py-2 text-sm text-gray-500">No matching models.</p>
                )}
              </div>
            )}
          </div>
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
