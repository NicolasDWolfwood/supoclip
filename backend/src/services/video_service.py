"""
Video service - handles video processing business logic.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import asyncio

from ..utils.async_helpers import run_in_thread
from ..youtube_utils import (
    download_youtube_video,
    get_youtube_video_title,
    get_youtube_video_id
)
from ..video_utils import (
    get_video_transcript,
    get_cached_formatted_transcript,
    create_clips_with_transitions
)
from ..ai import get_most_relevant_parts_by_transcript
from ..config import Config

logger = logging.getLogger(__name__)
config = Config()


class VideoService:
    """Service for video processing operations."""

    @staticmethod
    async def download_video(url: str, progress_callback: Optional[callable] = None) -> Optional[Path]:
        """
        Download a YouTube video asynchronously.
        Runs the sync download_youtube_video in a thread pool.
        """
        logger.info(f"Starting video download: {url}")
        loop = asyncio.get_running_loop()

        def on_download_progress(download_percent: int, message: str):
            if not progress_callback:
                return

            # Download stage occupies 10%-30% of overall progress.
            overall_progress = 10 + int((max(0, min(100, download_percent)) / 100) * 20)
            is_cached = "skipping download" in message.lower() or "found existing download" in message.lower()
            asyncio.run_coroutine_threadsafe(
                progress_callback(
                    overall_progress,
                    message,
                    {
                        "stage": "download",
                        "stage_progress": max(0, min(100, download_percent)),
                        "overall_progress": overall_progress,
                        "cached": is_cached,
                    },
                ),
                loop,
            )

        video_path = await run_in_thread(download_youtube_video, url, 3, on_download_progress)

        if not video_path:
            logger.error(f"Failed to download video: {url}")
            return None

        logger.info(f"Video downloaded successfully: {video_path}")
        return video_path

    @staticmethod
    async def get_video_title(url: str) -> str:
        """
        Get video title asynchronously.
        Returns a default title if retrieval fails.
        """
        try:
            title = await run_in_thread(get_youtube_video_title, url)
            return title or "YouTube Video"
        except Exception as e:
            logger.warning(f"Failed to get video title: {e}")
            return "YouTube Video"

    @staticmethod
    async def generate_transcript(video_path: Path) -> str:
        """
        Generate transcript from video using AssemblyAI.
        Runs in thread pool to avoid blocking.
        """
        logger.info(f"Generating transcript for: {video_path}")
        transcript = await run_in_thread(get_video_transcript, str(video_path))
        logger.info(f"Transcript generated: {len(transcript)} characters")
        return transcript

    @staticmethod
    async def generate_transcript_with_progress(video_path: Path, progress_callback: Optional[callable] = None) -> str:
        """
        Generate transcript and emit heartbeat progress while waiting for AssemblyAI.
        This prevents the UI from appearing stuck during long transcription calls.
        """
        cached_transcript = await run_in_thread(get_cached_formatted_transcript, str(video_path))
        if cached_transcript:
            logger.info(f"Using cached transcript for: {video_path.name}")
            if progress_callback:
                await progress_callback(
                    50,
                    "Found existing transcript, skipping transcription.",
                    {"stage": "transcript", "stage_progress": 100, "overall_progress": 50, "cached": True},
                )
            return cached_transcript

        heartbeat_task = None
        stop_heartbeat = asyncio.Event()

        async def heartbeat():
            # Transcript stage maps to overall progress range 30..50.
            overall = 31
            stage_progress = 5
            while not stop_heartbeat.is_set():
                if progress_callback:
                    await progress_callback(
                        min(overall, 49),
                        "Generating transcript...",
                        {
                            "stage": "transcript",
                            "stage_progress": min(stage_progress, 95),
                            "overall_progress": min(overall, 49),
                        },
                    )
                overall += 1
                stage_progress += 5
                try:
                    await asyncio.wait_for(stop_heartbeat.wait(), timeout=4)
                except asyncio.TimeoutError:
                    pass

        try:
            if progress_callback:
                heartbeat_task = asyncio.create_task(heartbeat())
            transcript = await VideoService.generate_transcript(video_path)
            return transcript
        finally:
            stop_heartbeat.set()
            if heartbeat_task:
                await heartbeat_task

    @staticmethod
    async def analyze_transcript(transcript: str) -> Any:
        """
        Analyze transcript with AI to find relevant segments.
        This is already async, no need to wrap.
        """
        logger.info("Starting AI analysis of transcript")
        relevant_parts = await get_most_relevant_parts_by_transcript(transcript)
        logger.info(f"AI analysis complete: {len(relevant_parts.most_relevant_segments)} segments found")
        return relevant_parts

    @staticmethod
    async def analyze_transcript_with_progress(
        transcript: str,
        progress_callback: Optional[callable] = None,
    ) -> Any:
        """
        Analyze transcript and emit heartbeat progress while waiting for the LLM call.
        This keeps UI progress moving during long AI analysis.
        """
        heartbeat_task = None
        stop_heartbeat = asyncio.Event()

        async def heartbeat():
            # Analysis stage maps to overall progress range 50..70.
            overall = 51
            stage_progress = 5
            while not stop_heartbeat.is_set():
                if progress_callback:
                    await progress_callback(
                        min(overall, 69),
                        "Analyzing content with AI...",
                        {
                            "stage": "analysis",
                            "stage_progress": min(stage_progress, 95),
                            "overall_progress": min(overall, 69),
                        },
                    )
                overall += 1
                stage_progress += 5
                try:
                    await asyncio.wait_for(stop_heartbeat.wait(), timeout=3)
                except asyncio.TimeoutError:
                    pass

        try:
            if progress_callback:
                heartbeat_task = asyncio.create_task(heartbeat())
            return await VideoService.analyze_transcript(transcript)
        finally:
            stop_heartbeat.set()
            if heartbeat_task:
                await heartbeat_task

    @staticmethod
    async def create_video_clips(
        video_path: Path,
        segments: List[Dict[str, Any]],
        font_family: str = "TikTokSans-Regular",
        font_size: int = 24,
        font_color: str = "#FFFFFF"
    ) -> Dict[str, Any]:
        """
        Create video clips from segments with transitions and subtitles.
        Runs in thread pool as video processing is CPU-intensive.
        """
        logger.info(f"Creating {len(segments)} video clips")
        clips_output_dir = Path(config.temp_dir) / "clips"
        clips_output_dir.mkdir(parents=True, exist_ok=True)
        render_diagnostics: Dict[str, Any] = {}

        clips_info = await run_in_thread(
            create_clips_with_transitions,
            video_path,
            segments,
            clips_output_dir,
            font_family,
            font_size,
            font_color,
            render_diagnostics,
        )

        logger.info(f"Successfully created {len(clips_info)} clips")
        return {"clips": clips_info, "diagnostics": render_diagnostics}

    @staticmethod
    def determine_source_type(url: str) -> str:
        """Determine if source is YouTube or uploaded file."""
        video_id = get_youtube_video_id(url)
        return "youtube" if video_id else "video_url"

    @staticmethod
    def validate_uploaded_video_path(url: str) -> Path:
        """
        Validate that uploaded video paths stay within the managed uploads directory.
        Prevents processing arbitrary local filesystem paths.
        """
        uploads_dir = (Path(config.temp_dir) / "uploads").resolve()
        candidate_path = Path(url).expanduser()
        resolved_path = candidate_path.resolve()

        # Ensure file exists and is inside uploads directory.
        if not resolved_path.exists() or not resolved_path.is_file():
            raise ValueError("Video file not found")

        try:
            resolved_path.relative_to(uploads_dir)
        except ValueError as exc:
            raise ValueError("Invalid uploaded video path") from exc

        return resolved_path

    @staticmethod
    async def process_video_complete(
        url: str,
        source_type: str,
        font_family: str = "TikTokSans-Regular",
        font_size: int = 24,
        font_color: str = "#FFFFFF",
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Complete video processing pipeline.
        Returns dict with segments and clips info.

        progress_callback: Optional function to call with progress updates
                          Signature: async def callback(progress: int, message: str)
        """
        try:
            # Step 1: Get video path (download or use existing)
            if progress_callback:
                await progress_callback(
                    10,
                    "Downloading video...",
                    {"stage": "download", "stage_progress": 0, "overall_progress": 10}
                )

            if source_type == "youtube":
                video_path = await VideoService.download_video(url, progress_callback=progress_callback)
                if not video_path:
                    raise Exception("Failed to download video")
            else:
                video_path = VideoService.validate_uploaded_video_path(url)

            # Step 2: Generate transcript
            if progress_callback:
                await progress_callback(
                    30,
                    "Generating transcript...",
                    {"stage": "transcript", "stage_progress": 0, "overall_progress": 30}
                )

            transcript = await VideoService.generate_transcript_with_progress(
                video_path,
                progress_callback=progress_callback,
            )

            # Step 3: AI analysis
            if progress_callback:
                await progress_callback(
                    50,
                    "Analyzing content with AI...",
                    {"stage": "analysis", "stage_progress": 0, "overall_progress": 50}
                )

            relevant_parts = await VideoService.analyze_transcript_with_progress(
                transcript,
                progress_callback=progress_callback,
            )

            # Step 4: Create clips
            if progress_callback:
                await progress_callback(
                    70,
                    "Creating video clips...",
                    {"stage": "clips", "stage_progress": 0, "overall_progress": 70}
                )

            segments_json = [
                {
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "text": segment.text,
                    "relevance_score": segment.relevance_score,
                    "reasoning": segment.reasoning
                }
                for segment in relevant_parts.most_relevant_segments
            ]

            clip_result = await VideoService.create_video_clips(
                video_path,
                segments_json,
                font_family,
                font_size,
                font_color
            )
            clips_info = clip_result.get("clips", [])
            clip_generation_diagnostics = clip_result.get("diagnostics", {})

            if progress_callback:
                await progress_callback(
                    100,
                    "Processing complete!",
                    {"stage": "finalizing", "stage_progress": 100, "overall_progress": 100}
                )

            return {
                "segments": segments_json,
                "clips": clips_info,
                "summary": relevant_parts.summary if relevant_parts else None,
                "key_topics": relevant_parts.key_topics if relevant_parts else None,
                "analysis_diagnostics": relevant_parts.diagnostics if relevant_parts else None,
                "clip_generation_diagnostics": clip_generation_diagnostics,
            }

        except Exception as e:
            logger.error(f"Error in video processing pipeline: {e}")
            raise
