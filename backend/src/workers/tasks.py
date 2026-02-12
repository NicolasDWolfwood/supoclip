"""
Worker tasks - background jobs processed by arq workers.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TaskCancelledError(Exception):
    """Raised when a task is explicitly cancelled by an admin action."""


async def process_video_task(
    ctx: Dict[str, Any],
    task_id: str,
    url: str,
    source_type: str,
    user_id: str,
    font_family: str = "TikTokSans-Regular",
    font_size: int = 24,
    font_color: str = "#FFFFFF",
    transitions_enabled: bool = False,
) -> Dict[str, Any]:
    """
    Background worker task to process a video.

    Args:
        ctx: arq context (provides Redis connection and other utilities)
        task_id: Task ID to update
        url: Video URL or file path
        source_type: "youtube" or "video_url"
        user_id: User ID who created the task
        font_family: Font family for subtitles
        font_size: Font size for subtitles
        font_color: Font color for subtitles
        transitions_enabled: Whether transition effects should be applied

    Returns:
        Dict with processing results
    """
    from ..database import AsyncSessionLocal
    from ..services.task_service import TaskService
    from ..workers.job_queue import JobQueue
    from ..workers.progress import ProgressTracker

    logger.info(f"Worker processing task {task_id}")

    # Create progress tracker
    progress = ProgressTracker(ctx['redis'], task_id)

    async with AsyncSessionLocal() as db:
        task_service = TaskService(db)

        try:
            async def ensure_not_cancelled() -> None:
                if await JobQueue.is_task_cancelled(task_id):
                    raise TaskCancelledError("Cancelled by admin action")

            await ensure_not_cancelled()

            # Progress callback
            async def update_progress(percent: int, message: str, metadata: Optional[Dict[str, Any]] = None):
                await ensure_not_cancelled()
                await progress.update(percent, message, metadata=metadata)
                logger.info(f"Task {task_id}: {percent}% - {message}")
                await ensure_not_cancelled()

            # Process the video
            result = await task_service.process_task(
                task_id=task_id,
                url=url,
                source_type=source_type,
                font_family=font_family,
                font_size=font_size,
                font_color=font_color,
                transitions_enabled=transitions_enabled,
                progress_callback=update_progress,
                cancel_check=ensure_not_cancelled,
            )

            logger.info(f"Task {task_id} completed successfully")
            await progress.complete()
            return result

        except TaskCancelledError as e:
            message = str(e)
            logger.info(f"Task {task_id} cancelled: {message}")
            await task_service.task_repo.update_task_status(
                db,
                task_id,
                "error",
                progress_message=message,
            )
            await progress.error(message)
            return {"task_id": task_id, "cancelled": True, "message": message}

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            await progress.error(str(e))
            # Error will be caught by arq and task status will be updated
            raise
        finally:
            await JobQueue.clear_task_cancelled(task_id)


# Worker configuration for arq
class WorkerSettings:
    """Configuration for arq worker."""

    from ..config import Config
    from arq.connections import RedisSettings

    config = Config()

    # Functions to run
    functions = [process_video_task]
    # Must match the exact Redis ZSET key used by ArqRedis.enqueue_job.
    queue_name = "arq:queue"

    # Redis settings from environment
    redis_settings = RedisSettings(
        host=config.redis_host,
        port=config.redis_port,
        database=0
    )

    # Retry settings
    max_tries = 3  # Retry failed jobs up to 3 times
    job_timeout = 3600  # 1 hour timeout for video processing

    # Worker pool settings (local transcription is CPU-heavy).
    max_jobs = config.worker_max_jobs
