"""
Task service - orchestrates task creation and processing workflow.
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional, Callable, Awaitable, List, Tuple
import logging
import asyncio
import re
from pathlib import Path

from ..repositories.task_repository import TaskRepository
from ..repositories.source_repository import SourceRepository
from ..repositories.clip_repository import ClipRepository
from ..repositories.draft_clip_repository import DraftClipRepository
from .video_service import VideoService
from .secret_service import SecretService
from .ai_model_catalog_service import list_models_for_provider
from ..config import Config
from ..video_utils import load_cached_transcript_data

logger = logging.getLogger(__name__)
config = Config()
SUPPORTED_AI_PROVIDERS = {"openai", "google", "anthropic", "zai"}
DEFAULT_AI_MODELS = {
    "openai": "gpt-5-mini",
    "google": "gemini-2.5-pro",
    "anthropic": "claude-4-sonnet",
    "zai": "glm-5",
}
SUPPORTED_ZAI_ROUTING_MODES = {"auto", "subscription", "metered"}
SUPPORTED_ZAI_KEY_PROFILES = {"subscription", "metered"}
DRAFT_MIN_DURATION_SECONDS = 3
DRAFT_MAX_DURATION_SECONDS = 180
TIMELINE_INCREMENT_SECONDS = 0.5
_TIMESTAMP_SECONDS_RE = re.compile(r"^\d+(?:\.\d+)?$")


class TaskService:
    """Service for task workflow orchestration."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.task_repo = TaskRepository()
        self.source_repo = SourceRepository()
        self.clip_repo = ClipRepository()
        self.draft_clip_repo = DraftClipRepository()
        self.video_service = VideoService()
        self.secret_service = SecretService()

    async def create_task_with_source(
        self,
        user_id: str,
        url: str,
        title: Optional[str] = None,
        font_family: str = "TikTokSans-Regular",
        font_size: int = 24,
        font_color: str = "#FFFFFF",
        subtitle_style: Optional[Dict[str, Any]] = None,
        transitions_enabled: bool = False,
        transcription_provider: str = "local",
        ai_provider: str = "openai",
        review_before_render_enabled: bool = True,
        timeline_editor_enabled: bool = True,
    ) -> str:
        """
        Create a new task with associated source.
        Returns the task ID.
        """
        if not await self.task_repo.user_exists(self.db, user_id):
            raise ValueError(f"User {user_id} not found")

        source_type = self.video_service.determine_source_type(url)

        if not title:
            if source_type == "youtube":
                title = await self.video_service.get_video_title(url)
            else:
                title = "Uploaded Video"

        source_id = await self.source_repo.create_source(
            self.db,
            source_type=source_type,
            title=title,
            url=url,
        )

        task_id = await self.task_repo.create_task(
            self.db,
            user_id=user_id,
            source_id=source_id,
            status="queued",
            font_family=font_family,
            font_size=font_size,
            font_color=font_color,
            subtitle_style=subtitle_style,
            transitions_enabled=transitions_enabled,
            transcription_provider=transcription_provider,
            ai_provider=ai_provider,
            review_before_render_enabled=review_before_render_enabled,
            timeline_editor_enabled=timeline_editor_enabled,
        )

        logger.info(f"Created task {task_id} for user {user_id}")
        return task_id

    @staticmethod
    def _env_ai_key_for_provider(provider: str) -> Optional[str]:
        normalized_provider = (provider or "").strip().lower()
        if normalized_provider == "openai":
            return (config.openai_api_key or "").strip() or None
        if normalized_provider == "google":
            return (config.google_api_key or "").strip() or None
        if normalized_provider == "anthropic":
            return (config.anthropic_api_key or "").strip() or None
        if normalized_provider == "zai":
            return (config.zai_api_key or "").strip() or None
        return None

    @staticmethod
    def _normalize_zai_routing_mode(value: Optional[str]) -> str:
        normalized = (value or "").strip().lower()
        if normalized not in SUPPORTED_ZAI_ROUTING_MODES:
            return "auto"
        return normalized

    @staticmethod
    def _normalize_text_for_compare(value: Optional[str]) -> str:
        return " ".join((value or "").split()).strip().lower()

    @staticmethod
    def _parse_timestamp_to_seconds_strict(raw_timestamp: Any) -> float:
        value = str(raw_timestamp or "").strip()
        if not value:
            raise ValueError("timestamp is required")

        parts = value.split(":")
        if len(parts) == 2:
            minute_text, second_text = parts
            if not (minute_text.isdigit() and _TIMESTAMP_SECONDS_RE.match(second_text)):
                raise ValueError(f"Invalid timestamp format: {value}")
            minutes = int(minute_text)
            seconds = float(second_text)
            if seconds >= 60:
                raise ValueError(f"Invalid timestamp format: {value}")
            return minutes * 60 + seconds

        if len(parts) == 3:
            hour_text, minute_text, second_text = parts
            if not (hour_text.isdigit() and minute_text.isdigit() and _TIMESTAMP_SECONDS_RE.match(second_text)):
                raise ValueError(f"Invalid timestamp format: {value}")
            hours = int(hour_text)
            minutes = int(minute_text)
            seconds = float(second_text)
            if minutes > 59 or seconds >= 60:
                raise ValueError(f"Invalid timestamp format: {value}")
            return hours * 3600 + minutes * 60 + seconds

        raise ValueError(f"Invalid timestamp format: {value}")

    @staticmethod
    def _snap_to_timeline_increment(seconds: float) -> float:
        snapped = round(max(0.0, float(seconds)) / TIMELINE_INCREMENT_SECONDS) * TIMELINE_INCREMENT_SECONDS
        return round(max(0.0, snapped), 3)

    @classmethod
    def _format_seconds_to_timestamp(cls, seconds: float) -> str:
        snapped_seconds = cls._snap_to_timeline_increment(seconds)
        whole_seconds = int(snapped_seconds)
        fractional = snapped_seconds - whole_seconds
        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        remainder_seconds = whole_seconds % 60
        if abs(fractional - 0.5) < 1e-6:
            second_token = f"{remainder_seconds:02d}.5"
        else:
            second_token = f"{remainder_seconds:02d}"
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{second_token}"
        return f"{minutes:02d}:{second_token}"

    def _validate_clip_window(
        self,
        start_time: str,
        end_time: str,
    ) -> tuple[float, float, float]:
        start_seconds = self._snap_to_timeline_increment(self._parse_timestamp_to_seconds_strict(start_time))
        end_seconds = self._snap_to_timeline_increment(self._parse_timestamp_to_seconds_strict(end_time))
        if start_seconds >= end_seconds:
            raise ValueError("start_time must be less than end_time")
        duration_seconds = end_seconds - start_seconds
        if duration_seconds < DRAFT_MIN_DURATION_SECONDS or duration_seconds > DRAFT_MAX_DURATION_SECONDS:
            raise ValueError(
                f"Clip duration must be between {DRAFT_MIN_DURATION_SECONDS}s and {DRAFT_MAX_DURATION_SECONDS}s"
            )
        return (
            start_seconds,
            end_seconds,
            round(duration_seconds, 3),
        )

    def _validate_non_overlapping_draft_windows(self, drafts: List[Dict[str, Any]]) -> None:
        windows: List[tuple[str, float, float]] = []
        for draft in drafts:
            if draft.get("is_deleted"):
                continue
            draft_id = str(draft.get("id") or "")
            start_seconds = self._parse_timestamp_to_seconds_strict(str(draft.get("start_time") or ""))
            end_seconds = self._parse_timestamp_to_seconds_strict(str(draft.get("end_time") or ""))
            windows.append((draft_id, start_seconds, end_seconds))

        windows.sort(key=lambda item: (item[1], item[2], item[0]))
        for index in range(1, len(windows)):
            previous_id, _previous_start, previous_end = windows[index - 1]
            current_id, current_start, _current_end = windows[index]
            if current_start < (previous_end - 1e-6):
                raise ValueError(f"Draft clips overlap: {previous_id} and {current_id}")

    @staticmethod
    def _extract_text_from_transcript_cache(video_path: Path, clip_start: float, clip_end: float) -> str:
        transcript_data = load_cached_transcript_data(video_path)
        if not transcript_data or not transcript_data.get("words"):
            return ""

        clip_start_ms = int(max(0.0, clip_start) * 1000)
        clip_end_ms = int(max(clip_start, clip_end) * 1000)
        matched_words: List[str] = []
        for word in transcript_data.get("words", []):
            word_text = str(word.get("text") or "").strip()
            if not word_text:
                continue
            word_start = int(word.get("start") or 0)
            word_end = int(word.get("end") or 0)
            if word_start < clip_end_ms and word_end > clip_start_ms:
                matched_words.append(word_text)

        return " ".join(matched_words).strip()

    async def get_user_zai_routing_mode(self, user_id: str) -> str:
        if not await self.task_repo.user_exists(self.db, user_id):
            raise ValueError(f"User {user_id} not found")
        return await self.task_repo.get_user_zai_routing_mode(self.db, user_id)

    async def set_user_zai_routing_mode(self, user_id: str, routing_mode: str) -> str:
        if not await self.task_repo.user_exists(self.db, user_id):
            raise ValueError(f"User {user_id} not found")
        normalized_mode = self._normalize_zai_routing_mode(routing_mode)
        return await self.task_repo.set_user_zai_routing_mode(self.db, user_id, normalized_mode)

    async def save_user_ai_profile_key(
        self,
        user_id: str,
        provider: str,
        profile_name: str,
        api_key: str,
    ) -> None:
        normalized_provider = (provider or "").strip().lower()
        normalized_profile = (profile_name or "").strip().lower()
        if normalized_provider not in SUPPORTED_AI_PROVIDERS:
            raise ValueError(f"Unsupported AI provider: {provider}")
        if normalized_profile not in SUPPORTED_ZAI_KEY_PROFILES:
            raise ValueError(f"Unsupported key profile: {profile_name}")
        if not await self.task_repo.user_exists(self.db, user_id):
            raise ValueError(f"User {user_id} not found")
        encrypted = self.secret_service.encrypt(api_key)
        await self.task_repo.set_user_ai_key_profile(
            self.db,
            user_id,
            normalized_provider,
            normalized_profile,
            encrypted,
        )

    async def clear_user_ai_profile_key(
        self,
        user_id: str,
        provider: str,
        profile_name: str,
    ) -> None:
        normalized_provider = (provider or "").strip().lower()
        normalized_profile = (profile_name or "").strip().lower()
        if normalized_provider not in SUPPORTED_AI_PROVIDERS:
            raise ValueError(f"Unsupported AI provider: {provider}")
        if normalized_profile not in SUPPORTED_ZAI_KEY_PROFILES:
            raise ValueError(f"Unsupported key profile: {profile_name}")
        if not await self.task_repo.user_exists(self.db, user_id):
            raise ValueError(f"User {user_id} not found")
        await self.task_repo.clear_user_ai_key_profile(
            self.db,
            user_id,
            normalized_provider,
            normalized_profile,
        )

    async def get_effective_user_ai_api_key_attempts(
        self,
        user_id: str,
        provider: str,
        zai_routing_mode: Optional[str] = None,
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        normalized_provider = (provider or "").strip().lower()
        if normalized_provider not in SUPPORTED_AI_PROVIDERS:
            raise ValueError(f"Unsupported AI provider: {provider}")
        if not await self.task_repo.user_exists(self.db, user_id):
            raise ValueError(f"User {user_id} not found")

        attempts: List[Dict[str, str]] = []
        seen_keys: set[str] = set()

        def append_attempt(label: str, key: Optional[str]) -> None:
            normalized_key = (key or "").strip()
            if not normalized_key:
                return
            if normalized_key in seen_keys:
                return
            seen_keys.add(normalized_key)
            attempts.append({"label": label, "key": normalized_key})

        if normalized_provider != "zai":
            stored_encrypted_ai_key = await self.task_repo.get_user_encrypted_ai_key(
                self.db,
                user_id,
                normalized_provider,
            )
            if stored_encrypted_ai_key:
                append_attempt("saved", self.secret_service.decrypt(stored_encrypted_ai_key))
            append_attempt("env", self._env_ai_key_for_provider(normalized_provider))
            return attempts, None

        if zai_routing_mode is None:
            resolved_mode = await self.task_repo.get_user_zai_routing_mode(self.db, user_id)
        else:
            resolved_mode = self._normalize_zai_routing_mode(zai_routing_mode)

        subscription_key_encrypted = await self.task_repo.get_user_ai_key_profile_encrypted(
            self.db,
            user_id,
            "zai",
            "subscription",
        )
        metered_key_encrypted = await self.task_repo.get_user_ai_key_profile_encrypted(
            self.db,
            user_id,
            "zai",
            "metered",
        )
        legacy_key_encrypted = await self.task_repo.get_user_encrypted_ai_key(
            self.db,
            user_id,
            "zai",
        )
        subscription_key = self.secret_service.decrypt(subscription_key_encrypted) if subscription_key_encrypted else None
        metered_key = self.secret_service.decrypt(metered_key_encrypted) if metered_key_encrypted else None
        legacy_key = self.secret_service.decrypt(legacy_key_encrypted) if legacy_key_encrypted else None
        env_key = self._env_ai_key_for_provider("zai")

        if resolved_mode == "subscription":
            append_attempt("subscription", subscription_key)
        elif resolved_mode == "metered":
            append_attempt("metered", metered_key)
        else:
            append_attempt("subscription", subscription_key)
            append_attempt("metered", metered_key)
            append_attempt("saved", legacy_key)
            append_attempt("env", env_key)

        return attempts, resolved_mode

    def _compute_completion_message(self, result: Dict[str, Any], clip_ids: List[str]) -> str:
        completion_message = "Complete!"
        if len(clip_ids) > 0:
            return completion_message

        analysis_diagnostics = result.get("analysis_diagnostics") or {}
        clip_diagnostics = result.get("clip_generation_diagnostics") or {}
        raw_segments = analysis_diagnostics.get("raw_segments")
        validated_segments = analysis_diagnostics.get("validated_segments")
        error_text = analysis_diagnostics.get("error")

        if error_text:
            return f"No clips generated: AI analysis failed ({error_text})"

        if validated_segments == 0:
            rejected_counts = analysis_diagnostics.get("rejected_counts") or {}
            human_labels = {
                "insufficient_text": "too little text",
                "identical_timestamps": "same start/end timestamp",
                "invalid_duration": "invalid duration",
                "too_short": "segment too short",
                "invalid_timestamp_format": "bad timestamp format",
            }
            reject_bits = []
            for key, label in human_labels.items():
                count = rejected_counts.get(key, 0)
                if count:
                    reject_bits.append(f"{label}: {count}")
            rejection_summary = " ".join(reject_bits) if reject_bits else "no valid segments met timing/quality checks."
            return (
                "No clips generated: transcript did not contain strong standalone moments "
                f"(hooks, value, emotion, complete thought, 10-45s). {rejection_summary}"
            )

        created_clips = clip_diagnostics.get("created_clips", 0)
        attempted_segments = clip_diagnostics.get("attempted_segments", validated_segments or 0)
        sample_failures = clip_diagnostics.get("failure_samples") or []
        if attempted_segments > 0 and created_clips == 0:
            sample_error = sample_failures[0].get("error") if sample_failures else "rendering error"
            return (
                f"No clips generated: AI found {validated_segments} clip-worthy segments, "
                f"but rendering failed for all {attempted_segments}. Example error: {sample_error}"
            )

        return (
            f"No clips generated: AI returned {raw_segments or 0} segments, "
            f"{validated_segments or 0} passed validation, but none were rendered successfully."
        )

    async def _persist_generated_clips(
        self,
        task_id: str,
        clips: List[Dict[str, Any]],
    ) -> List[str]:
        clip_ids: List[str] = []
        for i, clip_info in enumerate(clips):
            clip_id = await self.clip_repo.create_clip(
                self.db,
                task_id=task_id,
                filename=clip_info["filename"],
                file_path=clip_info["path"],
                start_time=clip_info["start_time"],
                end_time=clip_info["end_time"],
                duration=clip_info["duration"],
                text=clip_info.get("text"),
                relevance_score=clip_info.get("relevance_score", 0.0),
                reasoning=clip_info.get("reasoning"),
                clip_order=i + 1,
            )
            clip_ids.append(clip_id)

        await self.task_repo.update_task_clips(self.db, task_id, clip_ids)
        return clip_ids

    async def _resolve_processing_credentials(
        self,
        transcription_provider: str,
        ai_provider: str,
        ai_routing_mode: Optional[str],
        user_id: Optional[str],
    ) -> Tuple[Optional[str], str, Optional[str], Optional[str], List[str], List[str]]:
        assembly_api_key: Optional[str] = None
        if transcription_provider == "assemblyai":
            stored_encrypted_key = None
            if user_id:
                stored_encrypted_key = await self.task_repo.get_user_encrypted_assembly_key(self.db, user_id)
            if stored_encrypted_key:
                assembly_api_key = self.secret_service.decrypt(stored_encrypted_key)
            else:
                assembly_api_key = config.assembly_ai_api_key

        selected_ai_provider = (ai_provider or "openai").strip().lower()
        resolved_zai_routing_mode: Optional[str] = None
        ai_key_attempts: List[Dict[str, str]] = []
        if selected_ai_provider in SUPPORTED_AI_PROVIDERS and user_id:
            ai_key_attempts, resolved_zai_routing_mode = await self.get_effective_user_ai_api_key_attempts(
                user_id=user_id,
                provider=selected_ai_provider,
                zai_routing_mode=ai_routing_mode,
            )
        elif selected_ai_provider in SUPPORTED_AI_PROVIDERS:
            fallback_key = self._env_ai_key_for_provider(selected_ai_provider)
            if fallback_key:
                ai_key_attempts = [{"label": "env", "key": fallback_key}]

        ai_api_key = ai_key_attempts[0]["key"] if ai_key_attempts else None
        ai_api_key_fallbacks = [attempt["key"] for attempt in ai_key_attempts[1:]]
        ai_key_labels = [attempt["label"] for attempt in ai_key_attempts]

        return (
            assembly_api_key,
            selected_ai_provider,
            resolved_zai_routing_mode,
            ai_api_key,
            ai_api_key_fallbacks,
            ai_key_labels,
        )

    async def _process_review_enabled_analysis(
        self,
        task_id: str,
        url: str,
        source_type: str,
        transcription_provider: str,
        ai_provider: str,
        ai_model: Optional[str],
        ai_routing_mode: Optional[str],
        transcription_options: Optional[Dict[str, Any]],
        subtitle_style: Optional[Dict[str, Any]],
        progress_callback: Optional[Callable],
        cancel_check: Optional[Callable[[], Awaitable[None]]],
        user_id: Optional[str],
        update_progress: Callable[[int, str, Optional[Dict[str, Any]]], Awaitable[None]],
    ) -> Dict[str, Any]:
        (
            assembly_api_key,
            selected_ai_provider,
            resolved_zai_routing_mode,
            ai_api_key,
            ai_api_key_fallbacks,
            ai_key_labels,
        ) = await self._resolve_processing_credentials(
            transcription_provider=transcription_provider,
            ai_provider=ai_provider,
            ai_routing_mode=ai_routing_mode,
            user_id=user_id,
        )

        analysis_result = await self.video_service.process_video_analysis(
            url=url,
            source_type=source_type,
            transcription_provider=transcription_provider,
            assembly_api_key=assembly_api_key,
            ai_provider=selected_ai_provider,
            ai_api_key=ai_api_key,
            ai_api_key_fallbacks=ai_api_key_fallbacks,
            ai_key_labels=ai_key_labels,
            ai_routing_mode=resolved_zai_routing_mode,
            ai_model=ai_model,
            transcription_options=transcription_options,
            progress_callback=update_progress,
            cancel_check=cancel_check,
        )

        await self.task_repo.update_task_status(
            self.db,
            task_id,
            "processing",
            progress=95,
            progress_message="Saving draft clips...",
        )

        drafts_payload: List[Dict[str, Any]] = []
        for index, segment in enumerate(analysis_result.get("segments") or [], start=1):
            start_time = str(segment.get("start_time") or "00:00")
            end_time = str(segment.get("end_time") or "00:00")
            try:
                start_seconds, end_seconds, duration_seconds = self._validate_clip_window(start_time, end_time)
            except ValueError as validation_error:
                logger.warning(
                    "Skipping invalid draft segment for task %s (%s -> %s): %s",
                    task_id,
                    start_time,
                    end_time,
                    validation_error,
                )
                continue

            text_value = str(segment.get("text") or "").strip()
            drafts_payload.append(
                {
                    "clip_order": index,
                    "start_time": self._format_seconds_to_timestamp(start_seconds),
                    "end_time": self._format_seconds_to_timestamp(end_seconds),
                    "duration": duration_seconds,
                    "original_start_time": self._format_seconds_to_timestamp(start_seconds),
                    "original_end_time": self._format_seconds_to_timestamp(end_seconds),
                    "original_duration": duration_seconds,
                    "original_text": text_value,
                    "edited_text": text_value,
                    "relevance_score": float(segment.get("relevance_score") or 0.0),
                    "reasoning": segment.get("reasoning"),
                    "created_by_user": False,
                    "is_selected": True,
                    "is_deleted": False,
                    "edited_word_timings_json": None,
                }
            )

        await self.draft_clip_repo.replace_task_drafts(self.db, task_id, drafts_payload)
        await self.task_repo.update_task_status(
            self.db,
            task_id,
            "awaiting_review",
            progress=100,
            progress_message="Analysis complete. Review draft clips before rendering.",
        )

        return {
            "task_id": task_id,
            "drafts_count": len(drafts_payload),
            "segments": analysis_result.get("segments") or [],
            "summary": analysis_result.get("summary"),
            "key_topics": analysis_result.get("key_topics"),
            "final_status": "awaiting_review",
            "final_progress": 100,
            "final_message": "Analysis complete. Awaiting review.",
        }

    async def _process_non_review_pipeline(
        self,
        task_id: str,
        url: str,
        source_type: str,
        font_family: str,
        font_size: int,
        font_color: str,
        transitions_enabled: bool,
        transcription_provider: str,
        ai_provider: str,
        ai_model: Optional[str],
        ai_routing_mode: Optional[str],
        transcription_options: Optional[Dict[str, Any]],
        subtitle_style: Optional[Dict[str, Any]],
        progress_callback: Optional[Callable],
        cancel_check: Optional[Callable[[], Awaitable[None]]],
        user_id: Optional[str],
        update_progress: Callable[[int, str, Optional[Dict[str, Any]]], Awaitable[None]],
    ) -> Dict[str, Any]:
        (
            assembly_api_key,
            selected_ai_provider,
            resolved_zai_routing_mode,
            ai_api_key,
            ai_api_key_fallbacks,
            ai_key_labels,
        ) = await self._resolve_processing_credentials(
            transcription_provider=transcription_provider,
            ai_provider=ai_provider,
            ai_routing_mode=ai_routing_mode,
            user_id=user_id,
        )

        result = await self.video_service.process_video_complete(
            url=url,
            source_type=source_type,
            font_family=font_family,
            font_size=font_size,
            font_color=font_color,
            transitions_enabled=transitions_enabled,
            transcription_provider=transcription_provider,
            assembly_api_key=assembly_api_key,
            ai_provider=selected_ai_provider,
            ai_api_key=ai_api_key,
            ai_api_key_fallbacks=ai_api_key_fallbacks,
            ai_key_labels=ai_key_labels,
            ai_routing_mode=resolved_zai_routing_mode,
            ai_model=ai_model,
            transcription_options=transcription_options,
            subtitle_style=subtitle_style,
            progress_callback=update_progress,
            cancel_check=cancel_check,
        )

        await self.task_repo.update_task_status(
            self.db,
            task_id,
            "processing",
            progress=95,
            progress_message="Saving clips...",
        )

        await self.draft_clip_repo.delete_drafts_by_task(self.db, task_id)
        clip_ids = await self._persist_generated_clips(task_id, result.get("clips") or [])

        completion_message = self._compute_completion_message(result, clip_ids)
        await self.task_repo.update_task_status(
            self.db,
            task_id,
            "completed",
            progress=100,
            progress_message=completion_message,
        )

        logger.info(f"Task {task_id} completed successfully with {len(clip_ids)} clips")

        return {
            "task_id": task_id,
            "clips_count": len(clip_ids),
            "segments": result.get("segments") or [],
            "summary": result.get("summary"),
            "key_topics": result.get("key_topics"),
            "final_status": "completed",
            "final_progress": 100,
            "final_message": completion_message,
        }

    async def _render_from_drafts(
        self,
        task_id: str,
        url: str,
        source_type: str,
        font_family: str,
        font_size: int,
        font_color: str,
        transitions_enabled: bool,
        subtitle_style: Optional[Dict[str, Any]],
        cancel_check: Optional[Callable[[], Awaitable[None]]],
        update_progress: Callable[[int, str, Optional[Dict[str, Any]]], Awaitable[None]],
    ) -> Dict[str, Any]:
        await update_progress(10, "Loading approved draft clips...")

        drafts = await self.draft_clip_repo.get_drafts_by_task(self.db, task_id)
        selected_drafts = [draft for draft in drafts if draft.get("is_selected") and not draft.get("is_deleted")]
        if not selected_drafts:
            raise ValueError("Finalize requires at least one selected draft clip")

        selected_drafts.sort(
            key=lambda draft: (
                self._parse_timestamp_to_seconds_strict(str(draft.get("start_time") or "00:00")),
                int(draft.get("clip_order") or 0),
            )
        )

        await update_progress(15, "Preparing source media...")
        video_path = await self.video_service.resolve_video_path(url=url, source_type=source_type)

        rendered_segments: List[Dict[str, Any]] = []
        total_selected = len(selected_drafts)
        for index, draft in enumerate(selected_drafts, start=1):
            start_time = str(draft.get("start_time") or "").strip()
            end_time = str(draft.get("end_time") or "").strip()
            start_seconds, end_seconds, duration_seconds = self._validate_clip_window(start_time, end_time)

            original_text = str(draft.get("original_text") or "").strip()
            edited_text = str(draft.get("edited_text") or "").strip() or original_text
            if not edited_text:
                raise ValueError(f"Selected clip {draft.get('clip_order')} has empty subtitle text")

            text_was_edited = self._normalize_text_for_compare(edited_text) != self._normalize_text_for_compare(original_text)
            word_timings_override = None
            if text_was_edited:
                await update_progress(
                    20,
                    f"Aligning edited subtitles ({index}/{total_selected})...",
                    {
                        "stage": "analysis",
                        "stage_progress": int((index / total_selected) * 100),
                        "overall_progress": 20,
                    },
                )
                try:
                    word_timings_override = await self.video_service.align_edited_subtitle_words(
                        video_path=video_path,
                        clip_start=start_seconds,
                        clip_end=end_seconds,
                        edited_text=edited_text,
                    )
                except Exception as alignment_error:
                    raise ValueError(
                        f"Failed to align edited subtitles for clip {draft.get('clip_order')}: {alignment_error}"
                    ) from alignment_error

                await self.draft_clip_repo.update_draft_word_timings(
                    self.db,
                    task_id=task_id,
                    draft_id=str(draft["id"]),
                    word_timings=word_timings_override,
                )
            else:
                if draft.get("edited_word_timings_json") is not None:
                    await self.draft_clip_repo.update_draft_word_timings(
                        self.db,
                        task_id=task_id,
                        draft_id=str(draft["id"]),
                        word_timings=None,
                    )

            rendered_segments.append(
                {
                    "start_time": self._format_seconds_to_timestamp(start_seconds),
                    "end_time": self._format_seconds_to_timestamp(end_seconds),
                    "duration": duration_seconds,
                    "text": edited_text,
                    "relevance_score": float(draft.get("relevance_score") or 0.0),
                    "reasoning": draft.get("reasoning"),
                    "subtitle_word_timings": word_timings_override,
                }
            )

        await update_progress(65, "Rendering approved clips...")
        render_result = await self.video_service.render_video_segments(
            video_path=video_path,
            segments=rendered_segments,
            font_family=font_family,
            font_size=font_size,
            font_color=font_color,
            subtitle_style=subtitle_style,
            transitions_enabled=transitions_enabled,
            progress_callback=update_progress,
            cancel_check=cancel_check,
        )

        await self.task_repo.update_task_status(
            self.db,
            task_id,
            "processing",
            progress=95,
            progress_message="Saving clips...",
        )

        await self.clip_repo.delete_clips_by_task(self.db, task_id)
        clip_ids = await self._persist_generated_clips(task_id, render_result.get("clips") or [])

        completion_message = "Complete!" if clip_ids else "No clips were rendered from selected draft clips."
        await self.task_repo.update_task_status(
            self.db,
            task_id,
            "completed",
            progress=100,
            progress_message=completion_message,
        )

        return {
            "task_id": task_id,
            "clips_count": len(clip_ids),
            "segments": rendered_segments,
            "summary": None,
            "key_topics": None,
            "final_status": "completed",
            "final_progress": 100,
            "final_message": completion_message,
        }

    async def process_task(
        self,
        task_id: str,
        url: str,
        source_type: str,
        font_family: str = "TikTokSans-Regular",
        font_size: int = 24,
        font_color: str = "#FFFFFF",
        transitions_enabled: bool = False,
        transcription_provider: str = "local",
        ai_provider: str = "openai",
        ai_model: Optional[str] = None,
        ai_routing_mode: Optional[str] = None,
        transcription_options: Optional[Dict[str, Any]] = None,
        subtitle_style: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None,
        cancel_check: Optional[Callable[[], Awaitable[None]]] = None,
        user_id: Optional[str] = None,
        render_from_drafts: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a task.
        - default path: full one-pass processing (or analysis-only when review is enabled)
        - finalize path: render clips from reviewed drafts
        """
        try:
            logger.info(f"Starting processing for task {task_id} (render_from_drafts={render_from_drafts})")

            await self.task_repo.update_task_status(
                self.db,
                task_id,
                "processing",
                progress=0,
                progress_message="Starting...",
            )
            if cancel_check:
                await cancel_check()

            progress_lock = asyncio.Lock()

            async def update_progress(
                progress: int,
                message: str,
                metadata: Optional[Dict[str, Any]] = None,
            ) -> None:
                async with progress_lock:
                    if cancel_check:
                        await cancel_check()
                    await self.task_repo.update_task_status(
                        self.db,
                        task_id,
                        "processing",
                        progress=progress,
                        progress_message=message,
                    )
                    if progress_callback:
                        await progress_callback(progress, message, metadata)
                    if cancel_check:
                        await cancel_check()

            if render_from_drafts:
                return await self._render_from_drafts(
                    task_id=task_id,
                    url=url,
                    source_type=source_type,
                    font_family=font_family,
                    font_size=font_size,
                    font_color=font_color,
                    transitions_enabled=transitions_enabled,
                    subtitle_style=subtitle_style,
                    cancel_check=cancel_check,
                    update_progress=update_progress,
                )

            task_record = await self.task_repo.get_task_by_id(self.db, task_id)
            review_before_render_enabled = bool(
                (task_record or {}).get("review_before_render_enabled", True)
            )

            if review_before_render_enabled:
                return await self._process_review_enabled_analysis(
                    task_id=task_id,
                    url=url,
                    source_type=source_type,
                    transcription_provider=transcription_provider,
                    ai_provider=ai_provider,
                    ai_model=ai_model,
                    ai_routing_mode=ai_routing_mode,
                    transcription_options=transcription_options,
                    subtitle_style=subtitle_style,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                    user_id=user_id,
                    update_progress=update_progress,
                )

            return await self._process_non_review_pipeline(
                task_id=task_id,
                url=url,
                source_type=source_type,
                font_family=font_family,
                font_size=font_size,
                font_color=font_color,
                transitions_enabled=transitions_enabled,
                transcription_provider=transcription_provider,
                ai_provider=ai_provider,
                ai_model=ai_model,
                ai_routing_mode=ai_routing_mode,
                transcription_options=transcription_options,
                subtitle_style=subtitle_style,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                user_id=user_id,
                update_progress=update_progress,
            )

        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            await self.task_repo.update_task_status(
                self.db,
                task_id,
                "error",
                progress_message=str(e),
            )
            raise

    async def get_task_draft_clips(self, task_id: str) -> List[Dict[str, Any]]:
        return await self.draft_clip_repo.get_drafts_by_task(self.db, task_id)

    async def update_task_draft_clips(
        self,
        task_id: str,
        updates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not isinstance(updates, list) or not updates:
            raise ValueError("draft_clips must be a non-empty list")

        existing_by_id = await self.draft_clip_repo.get_draft_map_by_task(self.db, task_id)
        if not existing_by_id:
            raise ValueError("No draft clips found for task")

        normalized_updates: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        draft_state: Dict[str, Dict[str, Any]] = {
            draft_id: dict(existing)
            for draft_id, existing in existing_by_id.items()
        }

        for item in updates:
            if not isinstance(item, dict):
                raise ValueError("Each draft clip update must be an object")

            draft_id = str(item.get("id") or "").strip()
            if not draft_id:
                raise ValueError("Each draft clip update must include id")
            if draft_id in seen_ids:
                raise ValueError(f"Duplicate draft clip id in payload: {draft_id}")
            seen_ids.add(draft_id)

            existing = existing_by_id.get(draft_id)
            if not existing:
                raise ValueError(f"Draft clip not found: {draft_id}")

            start_time = str(item.get("start_time", existing["start_time"])).strip()
            end_time = str(item.get("end_time", existing["end_time"])).strip()
            start_seconds, end_seconds, duration_seconds = self._validate_clip_window(start_time, end_time)

            normalized_update: Dict[str, Any] = {
                "id": draft_id,
                "start_time": self._format_seconds_to_timestamp(start_seconds),
                "end_time": self._format_seconds_to_timestamp(end_seconds),
                "duration": duration_seconds,
            }

            if "edited_text" in item:
                if item.get("edited_text") is None:
                    normalized_update["edited_text"] = ""
                else:
                    normalized_update["edited_text"] = str(item.get("edited_text"))

            if "is_selected" in item:
                normalized_update["is_selected"] = bool(item.get("is_selected"))

            text_changed = (
                "edited_text" in normalized_update
                and self._normalize_text_for_compare(normalized_update["edited_text"])
                != self._normalize_text_for_compare(existing.get("edited_text"))
            )
            timing_changed = (
                normalized_update["start_time"] != str(existing.get("start_time"))
                or normalized_update["end_time"] != str(existing.get("end_time"))
            )
            if text_changed or timing_changed:
                normalized_update["edited_word_timings_json"] = None

            normalized_updates.append(normalized_update)
            draft_state[draft_id].update(normalized_update)

        self._validate_non_overlapping_draft_windows(list(draft_state.values()))

        await self.draft_clip_repo.bulk_update_drafts(self.db, task_id, normalized_updates)
        return await self.draft_clip_repo.get_drafts_by_task(self.db, task_id)

    async def create_task_draft_clip(
        self,
        task_id: str,
        start_time: str,
        end_time: str,
        source_url: str,
        source_type: str,
        edited_text: Optional[str] = None,
        is_selected: Optional[bool] = None,
    ) -> Dict[str, Any]:
        start_seconds, end_seconds, duration_seconds = self._validate_clip_window(start_time, end_time)
        normalized_start_time = self._format_seconds_to_timestamp(start_seconds)
        normalized_end_time = self._format_seconds_to_timestamp(end_seconds)

        existing_drafts = await self.draft_clip_repo.get_drafts_by_task(self.db, task_id)
        proposed_drafts = list(existing_drafts) + [
            {
                "id": "__new__",
                "start_time": normalized_start_time,
                "end_time": normalized_end_time,
                "is_deleted": False,
            }
        ]
        self._validate_non_overlapping_draft_windows(proposed_drafts)

        source_video_path = await self.video_service.resolve_video_path(url=source_url, source_type=source_type)
        transcript_text = self._extract_text_from_transcript_cache(
            source_video_path,
            start_seconds,
            end_seconds,
        )

        preferred_text = str(edited_text or "").strip()
        base_text = preferred_text or transcript_text
        clip_order = await self.draft_clip_repo.get_next_clip_order(self.db, task_id)

        payload = {
            "clip_order": clip_order,
            "start_time": normalized_start_time,
            "end_time": normalized_end_time,
            "duration": duration_seconds,
            "original_start_time": normalized_start_time,
            "original_end_time": normalized_end_time,
            "original_duration": duration_seconds,
            "original_text": base_text,
            "edited_text": preferred_text or base_text,
            "relevance_score": 0.0,
            "reasoning": "Added manually during review",
            "created_by_user": True,
            "is_selected": bool(is_selected) if is_selected is not None else bool(base_text),
            "is_deleted": False,
            "edited_word_timings_json": None,
        }
        created_id = await self.draft_clip_repo.create_draft(self.db, task_id, payload)

        draft_map = await self.draft_clip_repo.get_draft_map_by_task(self.db, task_id)
        draft = draft_map.get(created_id)
        if not draft:
            raise ValueError("Failed to create draft clip")
        return draft

    async def delete_task_draft_clip(self, task_id: str, draft_id: str) -> None:
        existing = await self.draft_clip_repo.get_draft_map_by_task(self.db, task_id)
        if draft_id not in existing:
            raise ValueError("Draft clip not found")
        await self.draft_clip_repo.soft_delete_draft(self.db, task_id=task_id, draft_id=draft_id)

    async def restore_task_draft_clips(self, task_id: str) -> List[Dict[str, Any]]:
        await self.draft_clip_repo.restore_task_drafts(self.db, task_id)
        restored = await self.draft_clip_repo.get_drafts_by_task(self.db, task_id)
        self._validate_non_overlapping_draft_windows(restored)
        return restored

    async def get_user_transcription_settings(self, user_id: str) -> Dict[str, Any]:
        if not await self.task_repo.user_exists(self.db, user_id):
            raise ValueError(f"User {user_id} not found")
        encrypted_key = await self.task_repo.get_user_encrypted_assembly_key(self.db, user_id)
        return {
            "has_assembly_key": bool(encrypted_key),
        }

    async def save_user_assembly_key(self, user_id: str, assembly_api_key: str) -> None:
        if not await self.task_repo.user_exists(self.db, user_id):
            raise ValueError(f"User {user_id} not found")
        encrypted = self.secret_service.encrypt(assembly_api_key)
        await self.task_repo.set_user_encrypted_assembly_key(self.db, user_id, encrypted)

    async def clear_user_assembly_key(self, user_id: str) -> None:
        if not await self.task_repo.user_exists(self.db, user_id):
            raise ValueError(f"User {user_id} not found")
        await self.task_repo.clear_user_encrypted_assembly_key(self.db, user_id)

    async def get_user_ai_settings(self, user_id: str) -> Dict[str, Any]:
        if not await self.task_repo.user_exists(self.db, user_id):
            raise ValueError(f"User {user_id} not found")
        result: Dict[str, Any] = {}
        for provider in SUPPORTED_AI_PROVIDERS:
            encrypted = await self.task_repo.get_user_encrypted_ai_key(self.db, user_id, provider)
            result[f"has_{provider}_key"] = bool(encrypted)
        zai_profiles = await self.task_repo.list_user_ai_key_profiles(self.db, user_id, "zai")
        result["has_zai_subscription_key"] = bool(zai_profiles.get("subscription"))
        result["has_zai_metered_key"] = bool(zai_profiles.get("metered"))
        result["zai_routing_mode"] = await self.task_repo.get_user_zai_routing_mode(self.db, user_id)
        result["has_zai_key"] = bool(
            result.get("has_zai_key")
            or result["has_zai_subscription_key"]
            or result["has_zai_metered_key"]
        )
        return result

    async def save_user_ai_key(self, user_id: str, provider: str, api_key: str) -> None:
        if provider not in SUPPORTED_AI_PROVIDERS:
            raise ValueError(f"Unsupported AI provider: {provider}")
        if not await self.task_repo.user_exists(self.db, user_id):
            raise ValueError(f"User {user_id} not found")
        encrypted = self.secret_service.encrypt(api_key)
        await self.task_repo.set_user_encrypted_ai_key(self.db, user_id, provider, encrypted)

    async def clear_user_ai_key(self, user_id: str, provider: str) -> None:
        if provider not in SUPPORTED_AI_PROVIDERS:
            raise ValueError(f"Unsupported AI provider: {provider}")
        if not await self.task_repo.user_exists(self.db, user_id):
            raise ValueError(f"User {user_id} not found")
        await self.task_repo.clear_user_encrypted_ai_key(self.db, user_id, provider)

    async def list_available_ai_models(
        self,
        user_id: str,
        provider: str,
        zai_routing_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_provider = (provider or "").strip().lower()
        key_attempts, resolved_routing_mode = await self.get_effective_user_ai_api_key_attempts(
            user_id=user_id,
            provider=normalized_provider,
            zai_routing_mode=zai_routing_mode,
        )
        api_key = key_attempts[0]["key"] if key_attempts else None
        if not api_key:
            routing_hint = f" (routing mode: {resolved_routing_mode})" if resolved_routing_mode else ""
            raise ValueError(
                f"{normalized_provider} selected but no API key is configured{routing_hint}. Save one in Settings."
            )

        models = await asyncio.to_thread(
            list_models_for_provider,
            normalized_provider,
            api_key,
        )
        default_model = DEFAULT_AI_MODELS[normalized_provider]
        return {
            "provider": normalized_provider,
            "models": models,
            "default_model": default_model,
            "count": len(models),
            "zai_routing_mode": resolved_routing_mode,
        }

    async def get_task_with_clips(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task details with all clips."""
        task = await self.task_repo.get_task_by_id(self.db, task_id)

        if not task:
            return None

        clips = await self.clip_repo.get_clips_by_task(self.db, task_id)
        task["clips"] = clips
        task["clips_count"] = len(clips)

        return task

    async def get_user_tasks(self, user_id: str, limit: int = 50) -> list[Dict[str, Any]]:
        """Get all tasks for a user."""
        return await self.task_repo.get_user_tasks(self.db, user_id, limit)

    async def delete_task(self, task_id: str) -> None:
        """Delete a task and all its associated clips."""
        await self.clip_repo.delete_clips_by_task(self.db, task_id)
        await self.draft_clip_repo.delete_drafts_by_task(self.db, task_id)
        await self.task_repo.delete_task(self.db, task_id)
        logger.info(f"Deleted task {task_id} and all associated clips")

    async def delete_all_user_tasks(self, user_id: str) -> int:
        """Delete all tasks that belong to a user."""
        deleted_count = await self.task_repo.delete_tasks_by_user(self.db, user_id)
        logger.info(f"Deleted all tasks for user {user_id}: {deleted_count}")
        return deleted_count
