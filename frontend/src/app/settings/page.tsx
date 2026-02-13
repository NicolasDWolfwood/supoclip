"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { AlertCircle, Settings } from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { useSession } from "@/lib/auth-client";
import { SettingsSaveStatus } from "./components/settings-save-status";
import { SettingsSectionFont } from "./components/settings-section-font";
import { SettingsSectionProcessing } from "./components/settings-section-processing";
import { SettingsSectionVideo } from "./components/settings-section-video";
import { SettingsSidebar } from "./components/settings-sidebar";
import {
  normalizeFontStyleOptions,
  normalizeFontWeight,
  normalizeLetterSpacing,
  normalizeLineHeight,
  normalizeShadowBlur,
  normalizeShadowOffset,
  normalizeShadowOpacity,
  normalizeStrokeWidth,
  type TextAlignOption,
  type TextTransformOption,
} from "@/lib/font-style-options";
import {
  arePreferencesEqual,
  DEFAULT_AI_MODELS,
  DEFAULT_USER_PREFERENCES,
  FALLBACK_AI_MODEL_OPTIONS,
  isAiProvider,
  isSettingsSection,
  isTranscriptionProvider,
  normalizeFontSize,
  SETTINGS_SECTION_META,
  SETTINGS_SECTIONS,
  type AiProvider,
  type SettingsSection,
  type UserPreferences,
} from "./settings-section-types";

interface SavePreferencesOptions {
  keepalive?: boolean;
}

function getActiveSection(sectionValue: string | null): SettingsSection {
  if (sectionValue === "processing") {
    return "ai";
  }
  return isSettingsSection(sectionValue) ? sectionValue : "font";
}

function SettingsPageContent() {
  const [preferencesDraft, setPreferencesDraft] = useState<UserPreferences>(DEFAULT_USER_PREFERENCES);
  const [lastSavedSnapshot, setLastSavedSnapshot] = useState<UserPreferences>(DEFAULT_USER_PREFERENCES);

  const [availableFonts, setAvailableFonts] = useState<Array<{ name: string; display_name: string }>>([]);
  const [isUploadingFont, setIsUploadingFont] = useState(false);
  const [fontUploadMessage, setFontUploadMessage] = useState<string | null>(null);
  const [fontUploadError, setFontUploadError] = useState<string | null>(null);

  const [isFetching, setIsFetching] = useState(true);
  const [isSavingPreferences, setIsSavingPreferences] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const [assemblyApiKey, setAssemblyApiKey] = useState("");
  const [hasSavedAssemblyKey, setHasSavedAssemblyKey] = useState(false);
  const [hasAssemblyEnvFallback, setHasAssemblyEnvFallback] = useState(false);
  const [isSavingAssemblyKey, setIsSavingAssemblyKey] = useState(false);
  const [assemblyKeyStatus, setAssemblyKeyStatus] = useState<string | null>(null);
  const [assemblyKeyError, setAssemblyKeyError] = useState<string | null>(null);

  const [aiApiKeys, setAiApiKeys] = useState<Record<AiProvider, string>>({
    openai: "",
    google: "",
    anthropic: "",
    zai: "",
  });
  const [hasSavedAiKeys, setHasSavedAiKeys] = useState<Record<AiProvider, boolean>>({
    openai: false,
    google: false,
    anthropic: false,
    zai: false,
  });
  const [hasEnvAiFallback, setHasEnvAiFallback] = useState<Record<AiProvider, boolean>>({
    openai: false,
    google: false,
    anthropic: false,
    zai: false,
  });
  const [isSavingAiKey, setIsSavingAiKey] = useState(false);
  const [aiKeyStatus, setAiKeyStatus] = useState<string | null>(null);
  const [aiKeyError, setAiKeyError] = useState<string | null>(null);

  const [aiModelOptions, setAiModelOptions] = useState<Record<AiProvider, string[]>>({
    openai: FALLBACK_AI_MODEL_OPTIONS.openai,
    google: FALLBACK_AI_MODEL_OPTIONS.google,
    anthropic: FALLBACK_AI_MODEL_OPTIONS.anthropic,
    zai: FALLBACK_AI_MODEL_OPTIONS.zai,
  });
  const [hasLoadedAiModels, setHasLoadedAiModels] = useState<Record<AiProvider, boolean>>({
    openai: false,
    google: false,
    anthropic: false,
    zai: false,
  });
  const [isLoadingAiModels, setIsLoadingAiModels] = useState(false);
  const [aiModelStatus, setAiModelStatus] = useState<string | null>(null);
  const [aiModelError, setAiModelError] = useState<string | null>(null);

  const activeAiProviderRef = useRef<AiProvider>(preferencesDraft.aiProvider);
  const latestAiModelsRequestRef = useRef(0);

  const { data: session, isPending } = useSession();
  const router = useRouter();
  const searchParams = useSearchParams();
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const activeSection = getActiveSection(searchParams.get("section"));
  const isDirty = useMemo(
    () => !arePreferencesEqual(preferencesDraft, lastSavedSnapshot),
    [preferencesDraft, lastSavedSnapshot],
  );

  const hasAiKeyForSelectedProvider =
    hasSavedAiKeys[preferencesDraft.aiProvider] || hasEnvAiFallback[preferencesDraft.aiProvider];

  const sectionNavItems = useMemo(
    () => SETTINGS_SECTIONS.map((section) => ({ id: section, ...SETTINGS_SECTION_META[section] })),
    [],
  );

  const loadFonts = useCallback(async () => {
    try {
      const response = await fetch(`${apiUrl}/fonts`);
      if (!response.ok) {
        return;
      }

      const data = await response.json();
      const fonts = data.fonts || [];
      setAvailableFonts(fonts);

      const fontFaceStyles = fonts
        .map((font: { name: string }) => {
          return `
            @font-face {
              font-family: '${font.name}';
              src: url('${apiUrl}/fonts/${font.name}') format('truetype');
              font-weight: normal;
              font-style: normal;
            }
          `;
        })
        .join("\n");

      const styleElement = document.createElement("style");
      styleElement.id = "custom-fonts";
      styleElement.innerHTML = fontFaceStyles;

      const existingStyle = document.getElementById("custom-fonts");
      if (existingStyle) {
        existingStyle.remove();
      }

      document.head.appendChild(styleElement);
    } catch (loadError) {
      console.error("Failed to load fonts:", loadError);
    }
  }, [apiUrl]);

  const fetchAiModels = useCallback(
    async (provider: AiProvider, options?: { showStatus?: boolean }): Promise<boolean> => {
      const showStatus = options?.showStatus ?? true;
      if (!session?.user?.id) {
        return false;
      }

      if (showStatus) {
        setAiModelStatus(null);
      }
      setAiModelError(null);
      setIsLoadingAiModels(true);
      latestAiModelsRequestRef.current += 1;
      const requestId = latestAiModelsRequestRef.current;

      try {
        const response = await fetch(`${apiUrl}/tasks/ai-settings/${provider}/models`, {
          headers: {
            user_id: session.user.id,
          },
        });

        const responseData = await response
          .json()
          .catch(() => ({} as { detail?: string; models?: unknown }));
        if (!response.ok) {
          throw new Error(responseData?.detail || `Failed to load ${provider} models`);
        }

        const rawModels = Array.isArray(responseData.models)
          ? responseData.models.filter((value: unknown): value is string => typeof value === "string")
          : [];
        const models: string[] = Array.from(
          new Set(rawModels.map((value: string) => value.trim()).filter((value: string) => value.length > 0)),
        );
        if (models.length === 0) {
          throw new Error(`No models were returned for ${provider}`);
        }

        setAiModelOptions((prev) => ({ ...prev, [provider]: models }));
        setHasLoadedAiModels((prev) => ({ ...prev, [provider]: true }));

        if (activeAiProviderRef.current === provider) {
          setPreferencesDraft((prev) => {
            if (prev.aiProvider !== provider) {
              return prev;
            }
            const trimmed = prev.aiModel.trim();
            if (trimmed && models.includes(trimmed)) {
              return prev;
            }
            if (models.includes(DEFAULT_AI_MODELS[provider])) {
              return { ...prev, aiModel: DEFAULT_AI_MODELS[provider] };
            }
            return { ...prev, aiModel: models[0] ?? DEFAULT_AI_MODELS[provider] };
          });

          if (showStatus) {
            setAiModelStatus(`Loaded ${models.length} models from ${provider}.`);
          }
        }
        return true;
      } catch (loadError) {
        const message = loadError instanceof Error ? loadError.message : `Failed to load ${provider} models`;
        setAiModelOptions((prev) => ({ ...prev, [provider]: FALLBACK_AI_MODEL_OPTIONS[provider] }));
        setHasLoadedAiModels((prev) => ({ ...prev, [provider]: false }));
        if (activeAiProviderRef.current === provider) {
          setAiModelError(message);
        }
        return false;
      } finally {
        if (latestAiModelsRequestRef.current === requestId) {
          setIsLoadingAiModels(false);
        }
      }
    },
    [apiUrl, session?.user?.id],
  );

  const savePreferences = useCallback(
    async (options?: SavePreferencesOptions): Promise<boolean> => {
      if (!session?.user?.id) {
        return false;
      }
      if (!isDirty) {
        return true;
      }

      setIsSavingPreferences(true);
      setSaveError(null);

      try {
        const resolvedAiModel = preferencesDraft.aiModel.trim() || DEFAULT_AI_MODELS[preferencesDraft.aiProvider];
        const payload: UserPreferences = {
          ...preferencesDraft,
          aiModel: resolvedAiModel,
        };

        const response = await fetch("/api/preferences", {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
          keepalive: options?.keepalive,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({} as { error?: string }));
          throw new Error(errorData.error || "Failed to save preferences");
        }

        setPreferencesDraft(payload);
        setLastSavedSnapshot(payload);
        setSaveError(null);
        return true;
      } catch (savePreferencesError) {
        console.error("Error saving preferences:", savePreferencesError);
        setSaveError(savePreferencesError instanceof Error ? savePreferencesError.message : "Failed to save preferences");
        return false;
      } finally {
        setIsSavingPreferences(false);
      }
    },
    [isDirty, preferencesDraft, session?.user?.id],
  );

  const updateSectionQueryParam = useCallback(
    (section: SettingsSection) => {
      const params = new URLSearchParams(searchParams.toString());
      params.set("section", section);
      router.replace(`/settings?${params.toString()}`);
    },
    [router, searchParams],
  );

  const handleSectionChange = useCallback(
    (section: SettingsSection) => {
      if (section === activeSection || isSavingPreferences) {
        return;
      }

      // Section changes should show persisted values, not unsaved draft edits.
      if (isDirty) {
        setPreferencesDraft({ ...lastSavedSnapshot });
        setSaveError(null);
      }

      updateSectionQueryParam(section);
    },
    [activeSection, isDirty, isSavingPreferences, lastSavedSnapshot, updateSectionQueryParam],
  );

  const saveAssemblyKey = useCallback(
    async (key: string): Promise<boolean> => {
      const trimmed = key.trim();
      if (!trimmed) {
        setAssemblyKeyError("AssemblyAI key cannot be empty.");
        return false;
      }

      setIsSavingAssemblyKey(true);
      setAssemblyKeyError(null);
      setAssemblyKeyStatus(null);

      try {
        const response = await fetch(`${apiUrl}/tasks/transcription-settings/assembly-key`, {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            user_id: session?.user?.id || "",
          },
          body: JSON.stringify({ assembly_api_key: trimmed }),
        });

        const responseData = await response.json().catch(() => ({} as { detail?: string }));
        if (!response.ok) {
          throw new Error(responseData?.detail || "Failed to save AssemblyAI key");
        }

        setHasSavedAssemblyKey(true);
        setAssemblyApiKey("");
        setAssemblyKeyStatus("AssemblyAI key saved.");
        return true;
      } catch (saveError) {
        const message = saveError instanceof Error ? saveError.message : "Failed to save AssemblyAI key";
        setAssemblyKeyError(message);
        return false;
      } finally {
        setIsSavingAssemblyKey(false);
      }
    },
    [apiUrl, session?.user?.id],
  );

  const deleteAssemblyKey = useCallback(async (): Promise<void> => {
    setIsSavingAssemblyKey(true);
    setAssemblyKeyError(null);
    setAssemblyKeyStatus(null);

    try {
      const response = await fetch(`${apiUrl}/tasks/transcription-settings/assembly-key`, {
        method: "DELETE",
        headers: {
          user_id: session?.user?.id || "",
        },
      });

      const responseData = await response.json().catch(() => ({} as { detail?: string }));
      if (!response.ok) {
        throw new Error(responseData?.detail || "Failed to remove AssemblyAI key");
      }

      setHasSavedAssemblyKey(false);
      setAssemblyApiKey("");
      setAssemblyKeyStatus("AssemblyAI key removed.");
    } catch (deleteError) {
      const message = deleteError instanceof Error ? deleteError.message : "Failed to remove AssemblyAI key";
      setAssemblyKeyError(message);
    } finally {
      setIsSavingAssemblyKey(false);
    }
  }, [apiUrl, session?.user?.id]);

  const saveAiProviderKey = useCallback(
    async (provider: AiProvider, key: string): Promise<boolean> => {
      const trimmed = key.trim();
      if (!trimmed) {
        setAiKeyError(`${provider} key cannot be empty.`);
        return false;
      }

      setIsSavingAiKey(true);
      setAiKeyError(null);
      setAiKeyStatus(null);

      try {
        const response = await fetch(`${apiUrl}/tasks/ai-settings/${provider}/key`, {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            user_id: session?.user?.id || "",
          },
          body: JSON.stringify({ api_key: trimmed }),
        });

        const responseData = await response.json().catch(() => ({} as { detail?: string }));
        if (!response.ok) {
          throw new Error(responseData?.detail || `Failed to save ${provider} key`);
        }

        setHasSavedAiKeys((prev) => ({ ...prev, [provider]: true }));
        setAiApiKeys((prev) => ({ ...prev, [provider]: "" }));
        setAiKeyStatus(`${provider} key saved.`);
        void fetchAiModels(provider);
        return true;
      } catch (saveError) {
        const message = saveError instanceof Error ? saveError.message : `Failed to save ${provider} key`;
        setAiKeyError(message);
        return false;
      } finally {
        setIsSavingAiKey(false);
      }
    },
    [apiUrl, fetchAiModels, session?.user?.id],
  );

  const deleteAiProviderKey = useCallback(
    async (provider: AiProvider): Promise<void> => {
      setIsSavingAiKey(true);
      setAiKeyError(null);
      setAiKeyStatus(null);

      try {
        const response = await fetch(`${apiUrl}/tasks/ai-settings/${provider}/key`, {
          method: "DELETE",
          headers: {
            user_id: session?.user?.id || "",
          },
        });

        const responseData = await response.json().catch(() => ({} as { detail?: string }));
        if (!response.ok) {
          throw new Error(responseData?.detail || `Failed to remove ${provider} key`);
        }

        setHasSavedAiKeys((prev) => ({ ...prev, [provider]: false }));
        setAiApiKeys((prev) => ({ ...prev, [provider]: "" }));
        setAiKeyStatus(`${provider} key removed.`);

        if (!hasEnvAiFallback[provider]) {
          setAiModelOptions((prev) => ({ ...prev, [provider]: FALLBACK_AI_MODEL_OPTIONS[provider] }));
          setHasLoadedAiModels((prev) => ({ ...prev, [provider]: false }));
          if (preferencesDraft.aiProvider === provider) {
            setAiModelStatus(null);
            setAiModelError(null);
          }
        }
      } catch (deleteError) {
        const message = deleteError instanceof Error ? deleteError.message : `Failed to remove ${provider} key`;
        setAiKeyError(message);
      } finally {
        setIsSavingAiKey(false);
      }
    },
    [apiUrl, hasEnvAiFallback, preferencesDraft.aiProvider, session?.user?.id],
  );

  const handleFontUpload = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) {
        return;
      }

      setFontUploadError(null);
      setFontUploadMessage(null);

      if (!file.name.toLowerCase().endsWith(".ttf")) {
        setFontUploadError("Only .ttf font files are supported.");
        event.target.value = "";
        return;
      }

      setIsUploadingFont(true);
      try {
        const formData = new FormData();
        formData.append("font", file);

        const response = await fetch(`${apiUrl}/fonts/upload`, {
          method: "POST",
          body: formData,
        });

        const responseData = await response
          .json()
          .catch(() => ({} as { detail?: string; message?: string; font?: { name?: string } }));

        if (!response.ok) {
          throw new Error(responseData?.detail || "Failed to upload font");
        }

        await loadFonts();

        if (typeof responseData?.font?.name === "string" && responseData.font.name.length > 0) {
          setPreferencesDraft((prev) => ({ ...prev, fontFamily: responseData.font?.name || prev.fontFamily }));
        }
        setFontUploadMessage(responseData?.message || "Font uploaded successfully.");
      } catch (uploadError) {
        setFontUploadError(uploadError instanceof Error ? uploadError.message : "Failed to upload font.");
      } finally {
        setIsUploadingFont(false);
        event.target.value = "";
      }
    },
    [apiUrl, loadFonts],
  );

  useEffect(() => {
    void loadFonts();
  }, [loadFonts]);

  useEffect(() => {
    if (!isPending && !session?.user?.id) {
      setIsFetching(false);
    }
  }, [isPending, session?.user?.id]);

  useEffect(() => {
    const sectionParam = searchParams.get("section");
    if (!isSettingsSection(sectionParam)) {
      updateSectionQueryParam("font");
    }
  }, [searchParams, updateSectionQueryParam]);

  useEffect(() => {
    if (!session?.user?.id) {
      return;
    }

    const loadPreferences = async () => {
      setIsFetching(true);
      try {
        const response = await fetch("/api/preferences");
        if (!response.ok) {
          return;
        }

        const data: Partial<UserPreferences> = await response.json();

        const resolvedAiProvider =
          typeof data.aiProvider === "string" && isAiProvider(data.aiProvider) ? data.aiProvider : "openai";

        const normalizedFontStyle = normalizeFontStyleOptions(data);

        const nextPreferences: UserPreferences = {
          ...normalizedFontStyle,
          transitionsEnabled: Boolean(data.transitionsEnabled),
          transcriptionProvider:
            typeof data.transcriptionProvider === "string" && isTranscriptionProvider(data.transcriptionProvider)
              ? data.transcriptionProvider
              : DEFAULT_USER_PREFERENCES.transcriptionProvider,
          aiProvider: resolvedAiProvider,
          aiModel:
            typeof data.aiModel === "string" && data.aiModel.trim().length > 0
              ? data.aiModel.trim()
              : DEFAULT_AI_MODELS[resolvedAiProvider],
        };

        setPreferencesDraft(nextPreferences);
        setLastSavedSnapshot(nextPreferences);
        setSaveError(null);
      } catch (loadError) {
        console.error("Failed to load preferences:", loadError);
      } finally {
        setIsFetching(false);
      }
    };

    void loadPreferences();
  }, [session?.user?.id]);

  useEffect(() => {
    activeAiProviderRef.current = preferencesDraft.aiProvider;
  }, [preferencesDraft.aiProvider]);

  useEffect(() => {
    if (!session?.user?.id) {
      return;
    }

    const loadTranscriptionSettings = async () => {
      try {
        const response = await fetch(`${apiUrl}/tasks/transcription-settings`, {
          headers: {
            user_id: session.user.id,
          },
        });
        if (!response.ok) {
          return;
        }

        const data = await response.json();
        setHasSavedAssemblyKey(Boolean(data.has_assembly_key));
        setHasAssemblyEnvFallback(Boolean(data.has_env_fallback));
      } catch (loadError) {
        console.error("Failed to load transcription settings:", loadError);
      }
    };

    void loadTranscriptionSettings();
  }, [apiUrl, session?.user?.id]);

  useEffect(() => {
    if (!session?.user?.id) {
      return;
    }

    const loadAiSettings = async () => {
      try {
        const response = await fetch(`${apiUrl}/tasks/ai-settings`, {
          headers: {
            user_id: session.user.id,
          },
        });
        if (!response.ok) {
          return;
        }

        const data = await response.json();
        setHasSavedAiKeys({
          openai: Boolean(data.has_openai_key),
          google: Boolean(data.has_google_key),
          anthropic: Boolean(data.has_anthropic_key),
          zai: Boolean(data.has_zai_key),
        });
        setHasEnvAiFallback({
          openai: Boolean(data.has_env_openai),
          google: Boolean(data.has_env_google),
          anthropic: Boolean(data.has_env_anthropic),
          zai: Boolean(data.has_env_zai),
        });
      } catch (loadError) {
        console.error("Failed to load AI settings:", loadError);
      }
    };

    void loadAiSettings();
  }, [apiUrl, session?.user?.id]);

  useEffect(() => {
    if (!session?.user?.id) {
      return;
    }

    const provider = preferencesDraft.aiProvider;
    const hasProviderKey = hasSavedAiKeys[provider] || hasEnvAiFallback[provider];
    if (!hasProviderKey) {
      setAiModelOptions((prev) => ({ ...prev, [provider]: FALLBACK_AI_MODEL_OPTIONS[provider] }));
      setHasLoadedAiModels((prev) => ({ ...prev, [provider]: false }));
      setAiModelStatus(null);
      setAiModelError(null);
      return;
    }

    void fetchAiModels(provider, { showStatus: false });
  }, [fetchAiModels, hasEnvAiFallback, hasSavedAiKeys, preferencesDraft.aiProvider, session?.user?.id]);

  useEffect(() => {
    if (!isDirty || isSavingPreferences) {
      return;
    }

    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
      void savePreferences({ keepalive: true });
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, [isDirty, isSavingPreferences, savePreferences]);

  if (isPending || isFetching) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center p-4">
        <div className="space-y-4">
          <Skeleton className="h-4 w-32 mx-auto" />
          <Skeleton className="h-4 w-48 mx-auto" />
          <Skeleton className="h-4 w-24 mx-auto" />
        </div>
      </div>
    );
  }

  if (!session?.user) {
    return (
      <div className="min-h-screen bg-white">
        <div className="max-w-4xl mx-auto px-4 py-24">
          <div className="text-center">
            <h1 className="text-3xl font-bold text-black mb-4">Sign In Required</h1>
            <p className="text-gray-600 mb-8">You need to sign in to access your settings</p>
            <Link href="/sign-in">
              <Button size="lg">Sign In</Button>
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white">
      <div className="border-b bg-white">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity cursor-pointer">
              <Image src="/brand/logo.png" alt="MrglSnips logo" width={96} height={96} className="h-24 w-24 object-contain" />
              <h1 className="text-xl font-bold text-black">MrglSnips</h1>
            </Link>

            <div className="flex items-center gap-3">
              <Avatar className="w-8 h-8">
                <AvatarImage src={session.user.image || ""} />
                <AvatarFallback className="bg-gray-100 text-black text-sm">
                  {session.user.name?.charAt(0) || session.user.email?.charAt(0) || "U"}
                </AvatarFallback>
              </Avatar>
              <div className="hidden sm:block">
                <p className="text-sm font-medium text-black">{session.user.name}</p>
                <p className="text-xs text-gray-500">{session.user.email}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-10">
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-2">
            <Settings className="w-6 h-6 text-black" />
            <h2 className="text-2xl font-bold text-black">Settings</h2>
          </div>
          <p className="text-gray-600">
            Configure your default preferences for video clip generation and per-user API keys.
          </p>
        </div>

        <div className="md:hidden mb-4 space-y-2">
          <Label className="text-sm font-medium text-black">Section</Label>
          <Select
            value={activeSection}
            onValueChange={(value) => {
              if (isSettingsSection(value)) {
                void handleSectionChange(value);
              }
            }}
            disabled={isSavingPreferences}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select section" />
            </SelectTrigger>
            <SelectContent>
              {SETTINGS_SECTIONS.map((section) => (
                <SelectItem key={section} value={section}>
                  {SETTINGS_SECTION_META[section].label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="grid md:grid-cols-[260px_1fr] gap-6">
          <SettingsSidebar
            sections={sectionNavItems}
            activeSection={activeSection}
            isSaving={isSavingPreferences}
            onSectionSelect={(section) => {
              void handleSectionChange(section);
            }}
          />

          <section className="rounded-lg border border-gray-200 bg-white p-4 sm:p-6">
            <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="text-sm font-semibold text-black">{SETTINGS_SECTION_META[activeSection].label}</p>
                <p className="text-xs text-gray-500">Use the Save button to apply default preference changes.</p>
              </div>
              <SettingsSaveStatus isDirty={isDirty} isSaving={isSavingPreferences} saveError={saveError} />
            </div>

            {saveError && (
              <Alert className="mb-4 border-red-200 bg-red-50">
                <AlertCircle className="h-4 w-4 text-red-500" />
                <AlertDescription className="text-sm text-red-700">{saveError}</AlertDescription>
              </Alert>
            )}

            {activeSection === "font" ? (
              <SettingsSectionFont
                isSaving={isSavingPreferences}
                availableFonts={availableFonts}
                fontFamily={preferencesDraft.fontFamily}
                fontSize={preferencesDraft.fontSize}
                fontColor={preferencesDraft.fontColor}
                fontWeight={preferencesDraft.fontWeight}
                lineHeight={preferencesDraft.lineHeight}
                letterSpacing={preferencesDraft.letterSpacing}
                textTransform={preferencesDraft.textTransform}
                textAlign={preferencesDraft.textAlign}
                strokeColor={preferencesDraft.strokeColor}
                strokeWidth={preferencesDraft.strokeWidth}
                shadowColor={preferencesDraft.shadowColor}
                shadowOpacity={preferencesDraft.shadowOpacity}
                shadowBlur={preferencesDraft.shadowBlur}
                shadowOffsetX={preferencesDraft.shadowOffsetX}
                shadowOffsetY={preferencesDraft.shadowOffsetY}
                isUploadingFont={isUploadingFont}
                fontUploadMessage={fontUploadMessage}
                fontUploadError={fontUploadError}
                onFontFamilyChange={(value) => {
                  setPreferencesDraft((prev) => ({ ...prev, fontFamily: value }));
                }}
                onFontSizeChange={(size) => {
                  setPreferencesDraft((prev) => ({ ...prev, fontSize: normalizeFontSize(size) }));
                }}
                onFontColorChange={(color) => {
                  setPreferencesDraft((prev) => ({ ...prev, fontColor: color }));
                }}
                onFontWeightChange={(weight) => {
                  setPreferencesDraft((prev) => ({ ...prev, fontWeight: normalizeFontWeight(weight) }));
                }}
                onLineHeightChange={(lineHeight) => {
                  setPreferencesDraft((prev) => ({ ...prev, lineHeight: normalizeLineHeight(lineHeight) }));
                }}
                onLetterSpacingChange={(spacing) => {
                  setPreferencesDraft((prev) => ({ ...prev, letterSpacing: normalizeLetterSpacing(spacing) }));
                }}
                onTextTransformChange={(textTransform) => {
                  setPreferencesDraft((prev) => ({ ...prev, textTransform: textTransform as TextTransformOption }));
                }}
                onTextAlignChange={(textAlign) => {
                  setPreferencesDraft((prev) => ({ ...prev, textAlign: textAlign as TextAlignOption }));
                }}
                onStrokeColorChange={(strokeColor) => {
                  setPreferencesDraft((prev) => ({ ...prev, strokeColor }));
                }}
                onStrokeWidthChange={(strokeWidth) => {
                  setPreferencesDraft((prev) => ({ ...prev, strokeWidth: normalizeStrokeWidth(strokeWidth) }));
                }}
                onShadowColorChange={(shadowColor) => {
                  setPreferencesDraft((prev) => ({ ...prev, shadowColor }));
                }}
                onShadowOpacityChange={(shadowOpacity) => {
                  setPreferencesDraft((prev) => ({ ...prev, shadowOpacity: normalizeShadowOpacity(shadowOpacity) }));
                }}
                onShadowBlurChange={(shadowBlur) => {
                  setPreferencesDraft((prev) => ({ ...prev, shadowBlur: normalizeShadowBlur(shadowBlur) }));
                }}
                onShadowOffsetXChange={(shadowOffsetX) => {
                  setPreferencesDraft((prev) => ({ ...prev, shadowOffsetX: normalizeShadowOffset(shadowOffsetX) }));
                }}
                onShadowOffsetYChange={(shadowOffsetY) => {
                  setPreferencesDraft((prev) => ({ ...prev, shadowOffsetY: normalizeShadowOffset(shadowOffsetY) }));
                }}
                onFontUpload={handleFontUpload}
              />
            ) : activeSection === "video" ? (
              <SettingsSectionVideo
                isSaving={isSavingPreferences}
                transitionsEnabled={preferencesDraft.transitionsEnabled}
                onToggleTransitions={() => {
                  setPreferencesDraft((prev) => ({ ...prev, transitionsEnabled: !prev.transitionsEnabled }));
                }}
              />
            ) : (
              <SettingsSectionProcessing
                isSaving={isSavingPreferences}
                transcriptionProvider={preferencesDraft.transcriptionProvider}
                aiProvider={preferencesDraft.aiProvider}
                aiModel={preferencesDraft.aiModel}
                aiModelOptions={aiModelOptions[preferencesDraft.aiProvider]}
                hasLoadedAiModels={hasLoadedAiModels[preferencesDraft.aiProvider]}
                hasAiKeyForSelectedProvider={hasAiKeyForSelectedProvider}
                isLoadingAiModels={isLoadingAiModels}
                isSavingAssemblyKey={isSavingAssemblyKey}
                isSavingAiKey={isSavingAiKey}
                assemblyApiKey={assemblyApiKey}
                hasSavedAssemblyKey={hasSavedAssemblyKey}
                hasAssemblyEnvFallback={hasAssemblyEnvFallback}
                assemblyKeyStatus={assemblyKeyStatus}
                assemblyKeyError={assemblyKeyError}
                aiApiKey={aiApiKeys[preferencesDraft.aiProvider]}
                hasSavedAiKey={hasSavedAiKeys[preferencesDraft.aiProvider]}
                hasEnvAiFallback={hasEnvAiFallback[preferencesDraft.aiProvider]}
                aiKeyStatus={aiKeyStatus}
                aiKeyError={aiKeyError}
                aiModelStatus={aiModelStatus}
                aiModelError={aiModelError}
                onTranscriptionProviderChange={(provider) => {
                  if (isTranscriptionProvider(provider)) {
                    setPreferencesDraft((prev) => ({ ...prev, transcriptionProvider: provider }));
                  }
                  setAssemblyKeyStatus(null);
                  setAssemblyKeyError(null);
                }}
                onAiProviderChange={(provider) => {
                  if (!isAiProvider(provider)) {
                    return;
                  }

                  setPreferencesDraft((prev) => {
                    const prevDefaultModel = DEFAULT_AI_MODELS[prev.aiProvider];
                    const trimmedModel = prev.aiModel.trim();
                    const nextModel =
                      !trimmedModel || trimmedModel === prevDefaultModel ? DEFAULT_AI_MODELS[provider] : prev.aiModel;

                    return {
                      ...prev,
                      aiProvider: provider,
                      aiModel: nextModel,
                    };
                  });

                  setAiKeyStatus(null);
                  setAiKeyError(null);
                  setAiModelStatus(null);
                  setAiModelError(null);
                }}
                onAiModelChange={(model) => {
                  setPreferencesDraft((prev) => ({ ...prev, aiModel: model }));
                }}
                onAssemblyApiKeyChange={setAssemblyApiKey}
                onSaveAssemblyKey={() => {
                  void saveAssemblyKey(assemblyApiKey);
                }}
                onDeleteAssemblyKey={() => {
                  void deleteAssemblyKey();
                }}
                onAiApiKeyChange={(value) => {
                  setAiApiKeys((prev) => ({ ...prev, [preferencesDraft.aiProvider]: value }));
                }}
                onSaveAiProviderKey={() => {
                  const provider = preferencesDraft.aiProvider;
                  void saveAiProviderKey(provider, aiApiKeys[provider]);
                }}
                onDeleteAiProviderKey={() => {
                  void deleteAiProviderKey(preferencesDraft.aiProvider);
                }}
                onRefreshAiModels={() => {
                  void fetchAiModels(preferencesDraft.aiProvider);
                }}
              />
            )}

            <div className="mt-6 flex justify-end">
              <Button
                type="button"
                variant={isDirty ? "default" : "outline"}
                className={
                  isDirty
                    ? "border-emerald-600 bg-emerald-600 text-white hover:bg-emerald-500"
                    : "border-gray-500/60 text-gray-200"
                }
                onClick={() => {
                  void savePreferences();
                }}
                disabled={isSavingPreferences || !isDirty}
              >
                {isSavingPreferences
                  ? "Saving..."
                  : activeSection === "font"
                    ? "Save Font Defaults"
                    : activeSection === "video"
                      ? "Save Video Defaults"
                      : "Save AI Defaults"}
              </Button>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

function SettingsPageFallback() {
  return (
    <div className="min-h-screen bg-white flex items-center justify-center p-4">
      <div className="space-y-4">
        <Skeleton className="h-4 w-32 mx-auto" />
        <Skeleton className="h-4 w-48 mx-auto" />
        <Skeleton className="h-4 w-24 mx-auto" />
      </div>
    </div>
  );
}

export default function SettingsPage() {
  return (
    <Suspense fallback={<SettingsPageFallback />}>
      <SettingsPageContent />
    </Suspense>
  );
}
