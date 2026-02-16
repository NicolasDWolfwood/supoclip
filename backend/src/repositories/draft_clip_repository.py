"""
Draft clip repository - handles all database operations for task_clip_drafts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession


class DraftClipRepository:
    """Repository for editable task draft clips."""

    @staticmethod
    async def replace_task_drafts(
        db: AsyncSession,
        task_id: str,
        drafts: List[Dict[str, Any]],
    ) -> List[str]:
        """Delete existing drafts for a task and insert a new ordered set."""
        await db.execute(
            sql_text("DELETE FROM task_clip_drafts WHERE task_id = :task_id"),
            {"task_id": task_id},
        )

        created_ids: List[str] = []
        for draft in drafts:
            result = await db.execute(
                sql_text(
                    """
                    INSERT INTO task_clip_drafts (
                        id,
                        task_id,
                        clip_order,
                        start_time,
                        end_time,
                        duration,
                        original_text,
                        edited_text,
                        relevance_score,
                        reasoning,
                        is_selected,
                        edited_word_timings_json,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        :id,
                        :task_id,
                        :clip_order,
                        :start_time,
                        :end_time,
                        :duration,
                        :original_text,
                        :edited_text,
                        :relevance_score,
                        :reasoning,
                        :is_selected,
                        :edited_word_timings_json,
                        NOW(),
                        NOW()
                    )
                    RETURNING id
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "task_id": task_id,
                    "clip_order": int(draft["clip_order"]),
                    "start_time": str(draft["start_time"]),
                    "end_time": str(draft["end_time"]),
                    "duration": float(draft["duration"]),
                    "original_text": draft.get("original_text"),
                    "edited_text": draft.get("edited_text"),
                    "relevance_score": float(draft["relevance_score"]),
                    "reasoning": draft.get("reasoning"),
                    "is_selected": bool(draft.get("is_selected", True)),
                    "edited_word_timings_json": draft.get("edited_word_timings_json"),
                },
            )
            created_ids.append(str(result.scalar()))

        await db.commit()
        return created_ids

    @staticmethod
    async def get_drafts_by_task(db: AsyncSession, task_id: str) -> List[Dict[str, Any]]:
        result = await db.execute(
            sql_text(
                """
                SELECT
                    id,
                    task_id,
                    clip_order,
                    start_time,
                    end_time,
                    duration,
                    original_text,
                    edited_text,
                    relevance_score,
                    reasoning,
                    is_selected,
                    edited_word_timings_json,
                    created_at,
                    updated_at
                FROM task_clip_drafts
                WHERE task_id = :task_id
                ORDER BY clip_order ASC
                """
            ),
            {"task_id": task_id},
        )

        drafts: List[Dict[str, Any]] = []
        for row in result.fetchall():
            drafts.append(
                {
                    "id": row.id,
                    "task_id": row.task_id,
                    "clip_order": row.clip_order,
                    "start_time": row.start_time,
                    "end_time": row.end_time,
                    "duration": float(row.duration),
                    "original_text": row.original_text,
                    "edited_text": row.edited_text,
                    "relevance_score": float(row.relevance_score),
                    "reasoning": row.reasoning,
                    "is_selected": bool(row.is_selected),
                    "edited_word_timings_json": row.edited_word_timings_json,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                }
            )

        return drafts

    @staticmethod
    async def get_draft_map_by_task(db: AsyncSession, task_id: str) -> Dict[str, Dict[str, Any]]:
        drafts = await DraftClipRepository.get_drafts_by_task(db, task_id)
        return {draft["id"]: draft for draft in drafts}

    @staticmethod
    async def bulk_update_drafts(
        db: AsyncSession,
        task_id: str,
        updates: List[Dict[str, Any]],
    ) -> None:
        for update in updates:
            draft_id = str(update["id"])
            set_clauses = ["updated_at = NOW()"]
            params: Dict[str, Any] = {
                "task_id": task_id,
                "draft_id": draft_id,
            }

            if "clip_order" in update:
                set_clauses.append("clip_order = :clip_order")
                params["clip_order"] = int(update["clip_order"])
            if "start_time" in update:
                set_clauses.append("start_time = :start_time")
                params["start_time"] = str(update["start_time"])
            if "end_time" in update:
                set_clauses.append("end_time = :end_time")
                params["end_time"] = str(update["end_time"])
            if "duration" in update:
                set_clauses.append("duration = :duration")
                params["duration"] = float(update["duration"])
            if "edited_text" in update:
                set_clauses.append("edited_text = :edited_text")
                params["edited_text"] = update.get("edited_text")
            if "is_selected" in update:
                set_clauses.append("is_selected = :is_selected")
                params["is_selected"] = bool(update["is_selected"])
            if "edited_word_timings_json" in update:
                set_clauses.append("edited_word_timings_json = :edited_word_timings_json")
                params["edited_word_timings_json"] = update.get("edited_word_timings_json")

            await db.execute(
                sql_text(
                    f"""
                    UPDATE task_clip_drafts
                    SET {', '.join(set_clauses)}
                    WHERE task_id = :task_id AND id = :draft_id
                    """
                ),
                params,
            )

        await db.commit()

    @staticmethod
    async def count_selected_drafts(db: AsyncSession, task_id: str) -> int:
        result = await db.execute(
            sql_text(
                """
                SELECT COUNT(*) AS count
                FROM task_clip_drafts
                WHERE task_id = :task_id
                  AND is_selected = true
                """
            ),
            {"task_id": task_id},
        )
        return int(result.scalar() or 0)

    @staticmethod
    async def delete_drafts_by_task(db: AsyncSession, task_id: str) -> int:
        result = await db.execute(
            sql_text("DELETE FROM task_clip_drafts WHERE task_id = :task_id"),
            {"task_id": task_id},
        )
        await db.commit()
        return int(result.rowcount or 0)

    @staticmethod
    async def update_draft_word_timings(
        db: AsyncSession,
        task_id: str,
        draft_id: str,
        word_timings: Optional[List[Dict[str, Any]]],
    ) -> None:
        await db.execute(
            sql_text(
                """
                UPDATE task_clip_drafts
                SET edited_word_timings_json = :word_timings,
                    updated_at = NOW()
                WHERE task_id = :task_id AND id = :draft_id
                """
            ),
            {
                "task_id": task_id,
                "draft_id": draft_id,
                "word_timings": word_timings,
            },
        )
        await db.commit()
