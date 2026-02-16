"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent } from "react";
import { Button } from "@/components/ui/button";

interface TimelineDraftClip {
  id: string;
  clip_order: number;
  start_time: string;
  end_time: string;
  is_selected: boolean;
}

interface DraftTimelineEditorProps {
  sourceVideoUrl: string | null;
  drafts: TimelineDraftClip[];
  disabled?: boolean;
  minDurationSeconds?: number;
  maxDurationSeconds?: number;
  onDraftTimingChange: (draftId: string, startTime: string, endTime: string) => void;
  onAddDraft: (startTime: string, endTime: string) => void;
}

type DragMode = "move" | "resize-start" | "resize-end";

interface DragState {
  clipId: string;
  mode: DragMode;
  pointerStartX: number;
  startSeconds: number;
  endSeconds: number;
  minStart: number;
  maxEnd: number;
}

const TIMELINE_INCREMENT_SECONDS = 0.5;
const SNAP_THRESHOLD_SECONDS = 0.5;
const DEFAULT_NEW_CLIP_SECONDS = 8;

function parseTimestampToSeconds(rawTimestamp: string): number {
  const value = (rawTimestamp || "").trim();
  if (!value) return 0;

  const parts = value.split(":");
  if (parts.length === 2) {
    const minutes = Number(parts[0]);
    const seconds = Number(parts[1]);
    if (!Number.isFinite(minutes) || !Number.isFinite(seconds)) return 0;
    return minutes * 60 + seconds;
  }
  if (parts.length === 3) {
    const hours = Number(parts[0]);
    const minutes = Number(parts[1]);
    const seconds = Number(parts[2]);
    if (!Number.isFinite(hours) || !Number.isFinite(minutes) || !Number.isFinite(seconds)) return 0;
    return hours * 3600 + minutes * 60 + seconds;
  }
  return Number.isFinite(Number(value)) ? Number(value) : 0;
}

function snapToIncrement(seconds: number): number {
  return Math.max(0, Math.round(seconds / TIMELINE_INCREMENT_SECONDS) * TIMELINE_INCREMENT_SECONDS);
}

function formatSecondsToTimestamp(rawSeconds: number): string {
  const seconds = snapToIncrement(rawSeconds);
  const totalWhole = Math.floor(seconds);
  const hasHalf = Math.abs(seconds - totalWhole - 0.5) < 1e-6;
  const hours = Math.floor(totalWhole / 3600);
  const minutes = Math.floor((totalWhole % 3600) / 60);
  const remainder = totalWhole % 60;
  const secondToken = hasHalf ? `${remainder.toString().padStart(2, "0")}.5` : remainder.toString().padStart(2, "0");
  if (hours > 0) {
    return `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${secondToken}`;
  }
  return `${minutes.toString().padStart(2, "0")}:${secondToken}`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function formatClock(seconds: number): string {
  const value = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(value / 60);
  const remaining = value % 60;
  return `${minutes.toString().padStart(2, "0")}:${remaining.toString().padStart(2, "0")}`;
}

export default function DraftTimelineEditor({
  sourceVideoUrl,
  drafts,
  disabled = false,
  minDurationSeconds = 3,
  maxDurationSeconds = 180,
  onDraftTimingChange,
  onAddDraft,
}: DraftTimelineEditorProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);
  const dragStateRef = useRef<DragState | null>(null);
  const [durationSeconds, setDurationSeconds] = useState(0);
  const [currentTimeSeconds, setCurrentTimeSeconds] = useState(0);
  const [activeClipId, setActiveClipId] = useState<string | null>(null);

  const draftWindows = useMemo(() => {
    return drafts
      .map((draft) => ({
        ...draft,
        startSeconds: snapToIncrement(parseTimestampToSeconds(draft.start_time)),
        endSeconds: snapToIncrement(parseTimestampToSeconds(draft.end_time)),
      }))
      .filter((draft) => draft.endSeconds > draft.startSeconds)
      .sort((a, b) => a.startSeconds - b.startSeconds || a.endSeconds - b.endSeconds || a.clip_order - b.clip_order);
  }, [drafts]);

  const getTrackWidth = useCallback(() => {
    return trackRef.current?.getBoundingClientRect().width || 0;
  }, []);

  const seekVideo = useCallback((seconds: number) => {
    if (!videoRef.current || !Number.isFinite(seconds)) return;
    videoRef.current.currentTime = clamp(seconds, 0, durationSeconds || seconds);
  }, [durationSeconds]);

  useEffect(() => {
    if (!videoRef.current) return;
    const video = videoRef.current;

    const handleLoadedMetadata = () => {
      setDurationSeconds(Number.isFinite(video.duration) ? Math.max(0, video.duration) : 0);
    };
    const handleTimeUpdate = () => {
      setCurrentTimeSeconds(video.currentTime || 0);
    };

    video.addEventListener("loadedmetadata", handleLoadedMetadata);
    video.addEventListener("timeupdate", handleTimeUpdate);
    return () => {
      video.removeEventListener("loadedmetadata", handleLoadedMetadata);
      video.removeEventListener("timeupdate", handleTimeUpdate);
    };
  }, [sourceVideoUrl]);

  const handleTimelineClick = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      if (!durationSeconds) return;
      const bounds = event.currentTarget.getBoundingClientRect();
      if (!bounds.width) return;
      const ratio = clamp((event.clientX - bounds.left) / bounds.width, 0, 1);
      const targetTime = ratio * durationSeconds;
      setCurrentTimeSeconds(targetTime);
      seekVideo(targetTime);
    },
    [durationSeconds, seekVideo],
  );

  const handlePointerMove = useCallback(
    (event: PointerEvent) => {
      const dragState = dragStateRef.current;
      if (!dragState || disabled || durationSeconds <= 0) return;

      const width = getTrackWidth();
      if (!width) return;

      const deltaSeconds = ((event.clientX - dragState.pointerStartX) / width) * durationSeconds;
      const clipLength = dragState.endSeconds - dragState.startSeconds;

      let nextStart = dragState.startSeconds;
      let nextEnd = dragState.endSeconds;

      if (dragState.mode === "move") {
        nextStart = clamp(dragState.startSeconds + deltaSeconds, dragState.minStart, dragState.maxEnd - clipLength);
        nextEnd = nextStart + clipLength;
      } else if (dragState.mode === "resize-start") {
        nextStart = clamp(
          dragState.startSeconds + deltaSeconds,
          dragState.minStart,
          dragState.endSeconds - minDurationSeconds,
        );
        nextEnd = dragState.endSeconds;
      } else {
        nextStart = dragState.startSeconds;
        nextEnd = clamp(
          dragState.endSeconds + deltaSeconds,
          dragState.startSeconds + minDurationSeconds,
          dragState.maxEnd,
        );
      }

      nextStart = snapToIncrement(nextStart);
      nextEnd = snapToIncrement(nextEnd);

      if (dragState.mode === "move") {
        const minStart = dragState.minStart;
        const maxStart = dragState.maxEnd - clipLength;
        nextStart = clamp(nextStart, minStart, maxStart);
        nextEnd = nextStart + clipLength;
        if (Math.abs(nextStart - dragState.minStart) <= SNAP_THRESHOLD_SECONDS) {
          nextStart = dragState.minStart;
          nextEnd = nextStart + clipLength;
        }
        if (Math.abs(nextEnd - dragState.maxEnd) <= SNAP_THRESHOLD_SECONDS) {
          nextEnd = dragState.maxEnd;
          nextStart = nextEnd - clipLength;
        }
      } else if (dragState.mode === "resize-start") {
        if (Math.abs(nextStart - dragState.minStart) <= SNAP_THRESHOLD_SECONDS) {
          nextStart = dragState.minStart;
        }
        nextStart = clamp(nextStart, dragState.minStart, dragState.endSeconds - minDurationSeconds);
      } else {
        if (Math.abs(nextEnd - dragState.maxEnd) <= SNAP_THRESHOLD_SECONDS) {
          nextEnd = dragState.maxEnd;
        }
        nextEnd = clamp(nextEnd, dragState.startSeconds + minDurationSeconds, dragState.maxEnd);
      }

      if (nextEnd - nextStart > maxDurationSeconds) {
        if (dragState.mode === "resize-start") {
          nextStart = nextEnd - maxDurationSeconds;
        } else if (dragState.mode === "resize-end") {
          nextEnd = nextStart + maxDurationSeconds;
        }
      }

      onDraftTimingChange(
        dragState.clipId,
        formatSecondsToTimestamp(nextStart),
        formatSecondsToTimestamp(nextEnd),
      );
    },
    [disabled, durationSeconds, getTrackWidth, maxDurationSeconds, minDurationSeconds, onDraftTimingChange],
  );

  const handlePointerUp = useCallback(() => {
    dragStateRef.current = null;
    setActiveClipId(null);
    window.removeEventListener("pointermove", handlePointerMove);
  }, [handlePointerMove]);

  const startDrag = useCallback(
    (
      event: ReactPointerEvent<HTMLDivElement>,
      clipId: string,
      mode: DragMode,
    ) => {
      if (disabled || durationSeconds <= 0) return;
      event.preventDefault();
      event.stopPropagation();

      const currentIndex = draftWindows.findIndex((clip) => clip.id === clipId);
      if (currentIndex === -1) return;
      const current = draftWindows[currentIndex];
      const previous = currentIndex > 0 ? draftWindows[currentIndex - 1] : null;
      const next = currentIndex < draftWindows.length - 1 ? draftWindows[currentIndex + 1] : null;

      dragStateRef.current = {
        clipId,
        mode,
        pointerStartX: event.clientX,
        startSeconds: current.startSeconds,
        endSeconds: current.endSeconds,
        minStart: previous ? previous.endSeconds : 0,
        maxEnd: next ? next.startSeconds : durationSeconds,
      };
      setActiveClipId(clipId);
      window.addEventListener("pointermove", handlePointerMove);
      window.addEventListener("pointerup", handlePointerUp, { once: true });
    },
    [disabled, draftWindows, durationSeconds, handlePointerMove, handlePointerUp],
  );

  useEffect(() => {
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [handlePointerMove, handlePointerUp]);

  const addClipDisabled = useMemo(() => {
    return disabled || !sourceVideoUrl || durationSeconds <= 0;
  }, [disabled, durationSeconds, sourceVideoUrl]);

  const handleAddAtPlayhead = useCallback(() => {
    if (addClipDisabled) return;

    const windows = draftWindows;
    const gaps: Array<{ start: number; end: number }> = [];
    let cursor = 0;
    for (const clip of windows) {
      if (clip.startSeconds > cursor + 1e-6) {
        gaps.push({ start: cursor, end: clip.startSeconds });
      }
      cursor = Math.max(cursor, clip.endSeconds);
    }
    if (durationSeconds > cursor + 1e-6) {
      gaps.push({ start: cursor, end: durationSeconds });
    }
    if (windows.length === 0) {
      gaps.push({ start: 0, end: durationSeconds });
    }

    const validGaps = gaps.filter((gap) => gap.end - gap.start >= minDurationSeconds);
    if (!validGaps.length) return;

    const playhead = clamp(currentTimeSeconds, 0, durationSeconds);
    let chosenGap = validGaps.find((gap) => playhead >= gap.start && playhead <= gap.end) || null;
    if (!chosenGap) {
      chosenGap = validGaps.reduce((best, candidate) => {
        const bestDistance = Math.min(Math.abs(playhead - best.start), Math.abs(playhead - best.end));
        const candidateDistance = Math.min(Math.abs(playhead - candidate.start), Math.abs(playhead - candidate.end));
        return candidateDistance < bestDistance ? candidate : best;
      }, validGaps[0]);
    }

    const gapDuration = chosenGap.end - chosenGap.start;
    const clipDuration = Math.min(DEFAULT_NEW_CLIP_SECONDS, maxDurationSeconds, gapDuration);
    const rawStart = clamp(playhead, chosenGap.start, chosenGap.end - clipDuration);
    const startSeconds = snapToIncrement(rawStart);
    const endSeconds = snapToIncrement(startSeconds + clipDuration);
    if (endSeconds - startSeconds < minDurationSeconds) return;
    onAddDraft(formatSecondsToTimestamp(startSeconds), formatSecondsToTimestamp(endSeconds));
  }, [
    addClipDisabled,
    currentTimeSeconds,
    draftWindows,
    durationSeconds,
    maxDurationSeconds,
    minDurationSeconds,
    onAddDraft,
  ]);

  return (
    <div className="space-y-3 rounded-md border border-gray-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm font-medium text-black">
          Source Timeline {durationSeconds > 0 ? `(${formatClock(currentTimeSeconds)} / ${formatClock(durationSeconds)})` : ""}
        </p>
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={handleAddAtPlayhead}
          disabled={addClipDisabled}
        >
          Add Clip At Playhead
        </Button>
      </div>

      {sourceVideoUrl ? (
        <video
          ref={videoRef}
          src={sourceVideoUrl}
          controls
          preload="metadata"
          className="w-full max-h-[360px] rounded-md bg-black"
        />
      ) : (
        <div className="rounded-md border border-dashed border-gray-300 p-4 text-xs text-gray-500">
          Source video is not ready for timeline editing.
        </div>
      )}

      <div
        ref={trackRef}
        className="relative h-20 w-full rounded-md border border-gray-200 bg-gray-50"
        onClick={handleTimelineClick}
      >
        {durationSeconds > 0 && (
          <div
            className="pointer-events-none absolute top-0 h-full w-px bg-red-500"
            style={{ left: `${(clamp(currentTimeSeconds, 0, durationSeconds) / durationSeconds) * 100}%` }}
          />
        )}
        {durationSeconds > 0 &&
          draftWindows.map((clip, displayIndex) => {
            const left = (clip.startSeconds / durationSeconds) * 100;
            const width = ((clip.endSeconds - clip.startSeconds) / durationSeconds) * 100;
            return (
              <div
                key={clip.id}
                className={`absolute top-6 h-8 rounded border text-[10px] text-white ${
                  clip.is_selected ? "border-blue-700 bg-blue-600/90" : "border-slate-500 bg-slate-500/90"
                } ${activeClipId === clip.id ? "ring-2 ring-amber-400" : ""}`}
                style={{ left: `${left}%`, width: `${Math.max(width, 1.2)}%` }}
                onPointerDown={(event) => startDrag(event, clip.id, "move")}
              >
                <div
                  className="absolute left-0 top-0 h-full w-2 cursor-ew-resize rounded-l bg-white/40"
                  onPointerDown={(event) => startDrag(event, clip.id, "resize-start")}
                />
                <div
                  className="absolute right-0 top-0 h-full w-2 cursor-ew-resize rounded-r bg-white/40"
                  onPointerDown={(event) => startDrag(event, clip.id, "resize-end")}
                />
                <button
                  type="button"
                  className="h-full w-full truncate px-2 text-left"
                  onClick={(event) => {
                    event.stopPropagation();
                    seekVideo(clip.startSeconds);
                    setCurrentTimeSeconds(clip.startSeconds);
                  }}
                >
                  Clip {displayIndex + 1}
                </button>
              </div>
            );
          })}
      </div>

      <p className="text-[11px] text-gray-500">
        Drag clip bodies to move or handles to resize. Snaps in 0.5s increments with no overlaps.
      </p>
    </div>
  );
}
