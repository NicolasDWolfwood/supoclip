import type { ChangeEvent } from "react";
import { Palette, Type } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";

interface SettingsSectionFontProps {
  isSaving: boolean;
  availableFonts: Array<{ name: string; display_name: string }>;
  fontFamily: string;
  fontSize: number;
  fontColor: string;
  isUploadingFont: boolean;
  fontUploadMessage: string | null;
  fontUploadError: string | null;
  onFontFamilyChange: (value: string) => void;
  onFontSizeChange: (size: number) => void;
  onFontColorChange: (color: string) => void;
  onFontUpload: (event: ChangeEvent<HTMLInputElement>) => void;
}

const SWATCH_COLORS = ["#FFFFFF", "#000000", "#FFD700", "#FF6B6B", "#4ECDC4", "#45B7D1"];

export function SettingsSectionFont({
  isSaving,
  availableFonts,
  fontFamily,
  fontSize,
  fontColor,
  isUploadingFont,
  fontUploadMessage,
  fontUploadError,
  onFontFamilyChange,
  onFontSizeChange,
  onFontColorChange,
  onFontUpload,
}: SettingsSectionFontProps) {
  return (
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
        <Select value={fontFamily} onValueChange={onFontFamilyChange} disabled={isSaving || isUploadingFont}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select font" />
          </SelectTrigger>
          <SelectContent>
            {availableFonts.map((font) => (
              <SelectItem key={font.name} value={font.name}>
                {font.display_name}
              </SelectItem>
            ))}
            {availableFonts.length === 0 && <SelectItem value="TikTokSans-Regular">TikTok Sans Regular</SelectItem>}
          </SelectContent>
        </Select>
        <input
          type="file"
          accept=".ttf,font/ttf"
          onChange={onFontUpload}
          disabled={isSaving || isUploadingFont}
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
            onValueChange={(value) => onFontSizeChange(value[0])}
            max={48}
            min={24}
            step={2}
            disabled={isSaving || isUploadingFont}
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
            onChange={(event) => onFontColorChange(event.target.value)}
            disabled={isSaving || isUploadingFont}
            className="w-12 h-10 rounded border border-gray-300 cursor-pointer disabled:cursor-not-allowed"
          />
          <Input
            type="text"
            value={fontColor}
            onChange={(event) => onFontColorChange(event.target.value)}
            disabled={isSaving || isUploadingFont}
            placeholder="#FFFFFF"
            className="flex-1 h-10"
            pattern="^#[0-9A-Fa-f]{6}$"
          />
        </div>
        <div className="flex gap-2 mt-2">
          {SWATCH_COLORS.map((color) => (
            <button
              key={color}
              type="button"
              onClick={() => onFontColorChange(color)}
              disabled={isSaving || isUploadingFont}
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
  );
}
