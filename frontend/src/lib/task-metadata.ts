const ACTIVE_TASK_STATUSES = new Set(["processing", "queued"]);

export function isHttpUrl(value: string | null | undefined): boolean {
  if (!value) return false;
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

export function formatSourceTypeLabel(sourceType?: string | null): string {
  if (!sourceType) return "Source";
  if (sourceType === "video_url") return "Video URL";
  return sourceType.charAt(0).toUpperCase() + sourceType.slice(1);
}

export function formatTaskRuntime(
  createdAt: string | null | undefined,
  updatedAt: string | null | undefined,
  status: string | null | undefined,
  nowMs: number = Date.now(),
): string {
  if (!createdAt) return "n/a";
  const started = new Date(createdAt);
  if (Number.isNaN(started.getTime())) return "n/a";

  const shouldUseNow = ACTIVE_TASK_STATUSES.has(status || "") || !updatedAt;
  const ended = shouldUseNow ? new Date(nowMs) : new Date(updatedAt);
  if (Number.isNaN(ended.getTime())) return "n/a";

  let totalSeconds = Math.max(0, Math.floor((ended.getTime() - started.getTime()) / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  totalSeconds %= 3600;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}
