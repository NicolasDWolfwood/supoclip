"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { useSession } from "@/lib/auth-client";
import Link from "next/link";
import Image from "next/image";
import {
  Type,
  Palette,
  CheckCircle,
  AlertCircle,
  Settings,
  Settings2,
  KeyRound,
  Cloud,
  Cpu,
  Bot,
} from "lucide-react";

const TRANSCRIPTION_PROVIDERS = ["local", "assemblyai"] as const;
const AI_PROVIDERS = ["openai", "google", "anthropic", "zai"] as const;

type TranscriptionProvider = (typeof TRANSCRIPTION_PROVIDERS)[number];
type AiProvider = (typeof AI_PROVIDERS)[number];

interface UserPreferences {
  fontFamily: string;
  fontSize: number;
  fontColor: string;
  transitionsEnabled: boolean;
  transcriptionProvider: TranscriptionProvider;
  aiProvider: AiProvider;
}

function normalizeFontSize(size: number): number {
  return Math.max(24, Math.min(48, size));
}

function isTranscriptionProvider(value: string): value is TranscriptionProvider {
  return TRANSCRIPTION_PROVIDERS.includes(value as TranscriptionProvider);
}

function isAiProvider(value: string): value is AiProvider {
  return AI_PROVIDERS.includes(value as AiProvider);
}

export default function SettingsPage() {
  const [fontFamily, setFontFamily] = useState("TikTokSans-Regular");
  const [fontSize, setFontSize] = useState(24);
  const [fontColor, setFontColor] = useState("#FFFFFF");
  const [transitionsEnabled, setTransitionsEnabled] = useState(false);
  const [transcriptionProvider, setTranscriptionProvider] = useState<TranscriptionProvider>("local");
  const [aiProvider, setAiProvider] = useState<AiProvider>("openai");

  const [availableFonts, setAvailableFonts] = useState<Array<{ name: string; display_name: string }>>([]);
  const [isUploadingFont, setIsUploadingFont] = useState(false);
  const [fontUploadMessage, setFontUploadMessage] = useState<string | null>(null);
  const [fontUploadError, setFontUploadError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isFetching, setIsFetching] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

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

  const { data: session, isPending } = useSession();
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

  useEffect(() => {
    void loadFonts();
  }, [loadFonts]);

  useEffect(() => {
    const loadPreferences = async () => {
      if (!session?.user?.id) {
        return;
      }

      setIsFetching(true);
      try {
        const response = await fetch("/api/preferences");
        if (response.ok) {
          const data: Partial<UserPreferences> = await response.json();
          setFontFamily(data.fontFamily || "TikTokSans-Regular");
          setFontSize(normalizeFontSize(data.fontSize || 24));
          setFontColor(data.fontColor || "#FFFFFF");
          setTransitionsEnabled(Boolean(data.transitionsEnabled));

          if (typeof data.transcriptionProvider === "string" && isTranscriptionProvider(data.transcriptionProvider)) {
            setTranscriptionProvider(data.transcriptionProvider);
          }

          if (typeof data.aiProvider === "string" && isAiProvider(data.aiProvider)) {
            setAiProvider(data.aiProvider);
          }
        }
      } catch (loadError) {
        console.error("Failed to load preferences:", loadError);
      } finally {
        setIsFetching(false);
      }
    };

    loadPreferences();
  }, [session?.user?.id]);

  useEffect(() => {
    const loadTranscriptionSettings = async () => {
      if (!session?.user?.id) {
        return;
      }

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

    loadTranscriptionSettings();
  }, [apiUrl, session?.user?.id]);

  useEffect(() => {
    const loadAiSettings = async () => {
      if (!session?.user?.id) {
        return;
      }

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

    loadAiSettings();
  }, [apiUrl, session?.user?.id]);

  const saveAssemblyKey = async (key: string): Promise<boolean> => {
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

      const responseData = await response.json().catch(() => ({}));
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
  };

  const deleteAssemblyKey = async (): Promise<void> => {
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

      const responseData = await response.json().catch(() => ({}));
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
  };

  const saveAiProviderKey = async (provider: AiProvider, key: string): Promise<boolean> => {
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

      const responseData = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(responseData?.detail || `Failed to save ${provider} key`);
      }

      setHasSavedAiKeys((prev) => ({ ...prev, [provider]: true }));
      setAiApiKeys((prev) => ({ ...prev, [provider]: "" }));
      setAiKeyStatus(`${provider} key saved.`);
      return true;
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : `Failed to save ${provider} key`;
      setAiKeyError(message);
      return false;
    } finally {
      setIsSavingAiKey(false);
    }
  };

  const deleteAiProviderKey = async (provider: AiProvider): Promise<void> => {
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

      const responseData = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(responseData?.detail || `Failed to remove ${provider} key`);
      }

      setHasSavedAiKeys((prev) => ({ ...prev, [provider]: false }));
      setAiApiKeys((prev) => ({ ...prev, [provider]: "" }));
      setAiKeyStatus(`${provider} key removed.`);
    } catch (deleteError) {
      const message = deleteError instanceof Error ? deleteError.message : `Failed to remove ${provider} key`;
      setAiKeyError(message);
    } finally {
      setIsSavingAiKey(false);
    }
  };

  const handleFontUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) {
      return;
    }

    setFontUploadError(null);
    setFontUploadMessage(null);

    if (!file.name.toLowerCase().endsWith(".ttf")) {
      setFontUploadError("Only .ttf font files are supported.");
      e.target.value = "";
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
        setFontFamily(responseData.font.name);
      }
      setFontUploadMessage(responseData?.message || "Font uploaded successfully.");
    } catch (uploadError) {
      setFontUploadError(uploadError instanceof Error ? uploadError.message : "Failed to upload font.");
    } finally {
      setIsUploadingFont(false);
      e.target.value = "";
    }
  };

  const handleSavePreferences = async () => {
    setIsLoading(true);
    setError(null);
    setSuccess(false);

    try {
      const response = await fetch("/api/preferences", {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          fontFamily,
          fontSize,
          fontColor,
          transitionsEnabled,
          transcriptionProvider,
          aiProvider,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to save preferences");
      }

      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (saveError) {
      console.error("Error saving preferences:", saveError);
      setError(saveError instanceof Error ? saveError.message : "Failed to save preferences");
    } finally {
      setIsLoading(false);
    }
  };

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

      <div className="max-w-4xl mx-auto px-4 py-16">
        <div className="max-w-xl mx-auto">
          <div className="mb-8">
            <div className="flex items-center gap-2 mb-2">
              <Settings className="w-6 h-6 text-black" />
              <h2 className="text-2xl font-bold text-black">Settings</h2>
            </div>
            <p className="text-gray-600">
              Configure your default preferences for video clip generation and per-user API keys.
            </p>
          </div>

          <Separator className="my-8" />

          <div className="space-y-8">
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-black mb-1">Default Font Settings</h3>
                <p className="text-sm text-gray-600">These defaults are applied to new video processing tasks.</p>
              </div>

              <div className="space-y-2">
                <Label className="text-sm font-medium text-black flex items-center gap-2">
                  <Type className="w-4 h-4" />
                  Font Family
                </Label>
                <Select value={fontFamily} onValueChange={setFontFamily} disabled={isLoading}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select font" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableFonts.map((font) => (
                      <SelectItem key={font.name} value={font.name}>
                        {font.display_name}
                      </SelectItem>
                    ))}
                    {availableFonts.length === 0 && (
                      <SelectItem value="TikTokSans-Regular">TikTok Sans Regular</SelectItem>
                    )}
                  </SelectContent>
                </Select>
                <input
                  type="file"
                  accept=".ttf,font/ttf"
                  onChange={handleFontUpload}
                  disabled={isLoading || isUploadingFont}
                  className="file:text-foreground placeholder:text-muted-foreground selection:bg-primary selection:text-primary-foreground dark:bg-input/30 border-input flex h-9 w-full min-w-0 rounded-md border bg-transparent px-3 py-1 text-sm shadow-xs transition-[color,box-shadow] outline-none file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive"
                />
                <p className="text-xs text-gray-500">
                  {isUploadingFont ? "Uploading font..." : "Upload a .ttf file to add it to this list."}
                </p>
                {fontUploadMessage && <p className="text-xs text-green-600">{fontUploadMessage}</p>}
                {fontUploadError && <p className="text-xs text-red-600">{fontUploadError}</p>}
              </div>

              <div className="space-y-2">
                <Label className="text-sm font-medium text-black">Font Size: {fontSize}px</Label>
                <div className="px-2 pt-5">
                  <Slider
                    value={[fontSize]}
                    onValueChange={(value) => setFontSize(normalizeFontSize(value[0]))}
                    max={48}
                    min={24}
                    step={2}
                    disabled={isLoading}
                    className="w-full"
                  />
                </div>
                <div className="flex justify-between text-xs text-gray-500">
                  <span>24px</span>
                  <span>48px</span>
                </div>
              </div>

              <div className="space-y-2">
                <Label className="text-sm font-medium text-black flex items-center gap-2">
                  <Palette className="w-4 h-4" />
                  Font Color
                </Label>
                <div className="flex items-center gap-2">
                  <input
                    type="color"
                    value={fontColor}
                    onChange={(e) => setFontColor(e.target.value)}
                    disabled={isLoading}
                    className="w-12 h-10 rounded border border-gray-300 cursor-pointer disabled:cursor-not-allowed"
                  />
                  <Input
                    type="text"
                    value={fontColor}
                    onChange={(e) => setFontColor(e.target.value)}
                    disabled={isLoading}
                    placeholder="#FFFFFF"
                    className="flex-1 h-10"
                    pattern="^#[0-9A-Fa-f]{6}$"
                  />
                </div>
                <div className="flex gap-2 mt-2">
                  {["#FFFFFF", "#000000", "#FFD700", "#FF6B6B", "#4ECDC4", "#45B7D1"].map((color) => (
                    <button
                      key={color}
                      type="button"
                      onClick={() => setFontColor(color)}
                      disabled={isLoading}
                      className="w-8 h-8 rounded border-2 border-gray-300 cursor-pointer hover:scale-110 transition-transform disabled:cursor-not-allowed"
                      style={{ backgroundColor: color }}
                      title={color}
                    />
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <Label className="text-sm font-medium text-black">Preview</Label>
                <div className="p-6 bg-black rounded-lg flex items-center justify-center min-h-[100px]">
                  <p
                    style={{
                      color: fontColor,
                      fontSize: `${Math.min(fontSize, 32)}px`,
                      fontFamily: `'${fontFamily}', system-ui, -apple-system, sans-serif`,
                      textAlign: "center",
                      lineHeight: "1.4",
                    }}
                    className="font-medium"
                  >
                    Your subtitle will look like this
                  </p>
                </div>
              </div>
            </div>

            <Separator />

            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Settings2 className="w-4 h-4 text-black" />
                <h3 className="text-lg font-semibold text-black">Default Processing Settings</h3>
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
                    onClick={() => setTransitionsEnabled((prev) => !prev)}
                    disabled={isLoading}
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
                  onValueChange={(value) => {
                    if (isTranscriptionProvider(value)) {
                      setTranscriptionProvider(value);
                    }
                    setAssemblyKeyStatus(null);
                    setAssemblyKeyError(null);
                  }}
                  disabled={isLoading || isSavingAssemblyKey}
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
                      onChange={(e) => setAssemblyApiKey(e.target.value ?? "")}
                      placeholder={
                        hasSavedAssemblyKey
                          ? "Saved key present (enter new key to replace)"
                          : "Paste your AssemblyAI key"
                      }
                      disabled={isLoading || isSavingAssemblyKey}
                    />
                    <div className="flex flex-wrap items-center gap-2">
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        disabled={isLoading || isSavingAssemblyKey || !assemblyApiKey.trim()}
                        onClick={() => {
                          void saveAssemblyKey(assemblyApiKey);
                        }}
                      >
                        {isSavingAssemblyKey ? "Saving..." : "Save Key"}
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        disabled={isLoading || isSavingAssemblyKey || !hasSavedAssemblyKey}
                        onClick={() => {
                          void deleteAssemblyKey();
                        }}
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
                  onValueChange={(value) => {
                    if (isAiProvider(value)) {
                      setAiProvider(value);
                    }
                    setAiKeyStatus(null);
                    setAiKeyError(null);
                  }}
                  disabled={isLoading || isSavingAiKey}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select AI provider" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="openai">OpenAI</SelectItem>
                    <SelectItem value="google">Google</SelectItem>
                    <SelectItem value="anthropic">Anthropic</SelectItem>
                    <SelectItem value="zai">z.ai (GLM)</SelectItem>
                  </SelectContent>
                </Select>

                <div className="space-y-2 rounded border border-gray-100 bg-gray-50 p-3">
                  <label htmlFor="ai-provider-key" className="text-xs font-medium text-black">
                    {aiProvider.toUpperCase()} API Key
                  </label>
                  <Input
                    id="ai-provider-key"
                    type="password"
                    value={aiApiKeys[aiProvider]}
                    onChange={(e) => setAiApiKeys((prev) => ({ ...prev, [aiProvider]: e.target.value ?? "" }))}
                    placeholder={
                      hasSavedAiKeys[aiProvider]
                        ? `Saved ${aiProvider} key present (enter new key to replace)`
                        : `Paste your ${aiProvider} API key`
                    }
                    disabled={isLoading || isSavingAiKey}
                  />
                  <div className="flex flex-wrap items-center gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      disabled={isLoading || isSavingAiKey || !(aiApiKeys[aiProvider] || "").trim()}
                      onClick={() => {
                        void saveAiProviderKey(aiProvider, aiApiKeys[aiProvider]);
                      }}
                    >
                      {isSavingAiKey ? "Saving..." : "Save Key"}
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      disabled={isLoading || isSavingAiKey || !hasSavedAiKeys[aiProvider]}
                      onClick={() => {
                        void deleteAiProviderKey(aiProvider);
                      }}
                    >
                      Remove Saved Key
                    </Button>
                    <span className="text-xs text-gray-500">
                      {hasSavedAiKeys[aiProvider]
                        ? "Saved key available"
                        : hasEnvAiFallback[aiProvider]
                          ? "No saved key; using backend env fallback"
                          : "No key configured"}
                    </span>
                  </div>
                  {aiKeyStatus && <p className="text-xs text-green-600">{aiKeyStatus}</p>}
                  {aiKeyError && <p className="text-xs text-red-600">{aiKeyError}</p>}
                </div>
              </div>
            </div>

            {success && (
              <Alert className="border-green-200 bg-green-50">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <AlertDescription className="text-sm text-green-700">Default preferences saved successfully!</AlertDescription>
              </Alert>
            )}

            {error && (
              <Alert className="border-red-200 bg-red-50">
                <AlertCircle className="h-4 w-4 text-red-500" />
                <AlertDescription className="text-sm text-red-700">{error}</AlertDescription>
              </Alert>
            )}

            <Button onClick={handleSavePreferences} disabled={isLoading} className="w-full h-11">
              {isLoading ? "Saving..." : "Save Preferences"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
