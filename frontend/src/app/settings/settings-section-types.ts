import {
  DEFAULT_FONT_STYLE_OPTIONS,
  normalizeFontSize,
  type FontStyleOptions,
} from "@/lib/font-style-options";

export const SETTINGS_SECTIONS = ["font", "processing"] as const;

export type SettingsSection = (typeof SETTINGS_SECTIONS)[number];

export const TRANSCRIPTION_PROVIDERS = ["local", "assemblyai"] as const;
export const AI_PROVIDERS = ["openai", "google", "anthropic", "zai"] as const;

export type TranscriptionProvider = (typeof TRANSCRIPTION_PROVIDERS)[number];
export type AiProvider = (typeof AI_PROVIDERS)[number];

export interface UserPreferences extends FontStyleOptions {
  transitionsEnabled: boolean;
  transcriptionProvider: TranscriptionProvider;
  aiProvider: AiProvider;
  aiModel: string;
}

export const DEFAULT_USER_PREFERENCES: UserPreferences = {
  ...DEFAULT_FONT_STYLE_OPTIONS,
  transitionsEnabled: false,
  transcriptionProvider: "local",
  aiProvider: "openai",
  aiModel: "gpt-5",
};

export const DEFAULT_AI_MODELS: Record<AiProvider, string> = {
  openai: "gpt-5",
  google: "gemini-2.5-pro",
  anthropic: "claude-4-sonnet",
  zai: "glm-5",
};

export const FALLBACK_AI_MODEL_OPTIONS: Record<AiProvider, string[]> = {
  openai: ["gpt-5", "gpt-5-mini", "gpt-4.1"],
  google: ["gemini-2.5-pro", "gemini-2.5-flash"],
  anthropic: ["claude-4-sonnet", "claude-3-5-haiku"],
  zai: ["glm-5"],
};

export const SETTINGS_SECTION_META: Record<SettingsSection, { label: string; description: string }> = {
  font: {
    label: "Default Font Settings",
    description: "Defaults for subtitle style on new tasks.",
  },
  processing: {
    label: "Default Processing Settings",
    description: "Providers, models, transitions, and API keys.",
  },
};

export function isTranscriptionProvider(value: string): value is TranscriptionProvider {
  return TRANSCRIPTION_PROVIDERS.includes(value as TranscriptionProvider);
}

export function isAiProvider(value: string): value is AiProvider {
  return AI_PROVIDERS.includes(value as AiProvider);
}

export function isSettingsSection(value: string | null): value is SettingsSection {
  return value !== null && SETTINGS_SECTIONS.includes(value as SettingsSection);
}

export function arePreferencesEqual(a: UserPreferences, b: UserPreferences): boolean {
  return (
    a.fontFamily === b.fontFamily &&
    a.fontSize === b.fontSize &&
    a.fontColor === b.fontColor &&
    a.fontWeight === b.fontWeight &&
    a.lineHeight === b.lineHeight &&
    a.letterSpacing === b.letterSpacing &&
    a.textTransform === b.textTransform &&
    a.textAlign === b.textAlign &&
    a.strokeColor === b.strokeColor &&
    a.strokeWidth === b.strokeWidth &&
    a.shadowColor === b.shadowColor &&
    a.shadowOpacity === b.shadowOpacity &&
    a.shadowBlur === b.shadowBlur &&
    a.shadowOffsetX === b.shadowOffsetX &&
    a.shadowOffsetY === b.shadowOffsetY &&
    a.transitionsEnabled === b.transitionsEnabled &&
    a.transcriptionProvider === b.transcriptionProvider &&
    a.aiProvider === b.aiProvider &&
    a.aiModel === b.aiModel
  );
}

export { normalizeFontSize };
