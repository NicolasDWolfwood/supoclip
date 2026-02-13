import os

from dotenv import load_dotenv
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://localhost:5432/supoclip"
)

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL query logging
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# Create async session maker
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Base class for all models
class Base(DeclarativeBase):
    pass

# Dependency to get database session
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Initialize database
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                """
                ALTER TABLE tasks
                ADD COLUMN IF NOT EXISTS transcription_provider VARCHAR(20) NOT NULL DEFAULT 'local'
                """
            )
        )
        await conn.execute(
            text(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_constraint
                        WHERE conname = 'check_tasks_transcription_provider'
                    ) THEN
                        ALTER TABLE tasks
                        ADD CONSTRAINT check_tasks_transcription_provider
                        CHECK (transcription_provider IN ('local', 'assemblyai'));
                    END IF;
                END $$;
                """
            )
        )
        await conn.execute(
            text(
                """
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS assembly_api_key_encrypted TEXT
                """
            )
        )
        await conn.execute(
            text(
                """
                ALTER TABLE tasks
                ADD COLUMN IF NOT EXISTS ai_provider VARCHAR(20) NOT NULL DEFAULT 'openai'
                """
            )
        )
        await conn.execute(
            text(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1
                        FROM pg_constraint
                        WHERE conname = 'check_tasks_ai_provider'
                    ) THEN
                        ALTER TABLE tasks DROP CONSTRAINT check_tasks_ai_provider;
                    END IF;
                    ALTER TABLE tasks
                    ADD CONSTRAINT check_tasks_ai_provider
                    CHECK (ai_provider IN ('openai', 'google', 'anthropic', 'zai'));
                END $$;
                """
            )
        )
        await conn.execute(
            text(
                """
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS openai_api_key_encrypted TEXT
                """
            )
        )
        await conn.execute(
            text(
                """
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS google_api_key_encrypted TEXT
                """
            )
        )
        await conn.execute(
            text(
                """
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS anthropic_api_key_encrypted TEXT
                """
            )
        )
        await conn.execute(
            text(
                """
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS zai_api_key_encrypted TEXT
                """
            )
        )
        await conn.execute(
            text(
                """
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS default_transitions_enabled BOOLEAN NOT NULL DEFAULT false
                """
            )
        )
        await conn.execute(
            text(
                """
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS default_transcription_provider VARCHAR(20) NOT NULL DEFAULT 'local'
                """
            )
        )
        await conn.execute(
            text(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_constraint
                        WHERE conname = 'check_users_default_transcription_provider'
                    ) THEN
                        ALTER TABLE users
                        ADD CONSTRAINT check_users_default_transcription_provider
                        CHECK (default_transcription_provider IN ('local', 'assemblyai'));
                    END IF;
                END $$;
                """
            )
        )
        await conn.execute(
            text(
                """
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS default_ai_provider VARCHAR(20) NOT NULL DEFAULT 'openai'
                """
            )
        )
        await conn.execute(
            text(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1
                        FROM pg_constraint
                        WHERE conname = 'check_users_default_ai_provider'
                    ) THEN
                        ALTER TABLE users DROP CONSTRAINT check_users_default_ai_provider;
                    END IF;
                    ALTER TABLE users
                    ADD CONSTRAINT check_users_default_ai_provider
                    CHECK (default_ai_provider IN ('openai', 'google', 'anthropic', 'zai'));
                END $$;
                """
            )
        )

# Close database connections
async def close_db():
    await engine.dispose()
