# Configuration Reference

This is the single source of truth for SupoClip runtime environment variables.

## Core Variables

| Variable | Required | Default | Used By | Notes |
|---|---|---|---|---|
| `ASSEMBLY_AI_API_KEY` | Yes | - | backend, worker | Required for transcription. |
| `LLM` | No | `openai:gpt-5-mini` | backend, worker | Primary model selector (`provider:model`). |
| `OPENAI_API_KEY` | Conditional | - | backend, worker | Required when `LLM` uses `openai:*`. |
| `GOOGLE_API_KEY` | Conditional | - | backend, worker | Required when `LLM` uses `google:*`. |
| `ANTHROPIC_API_KEY` | Conditional | - | backend, worker | Required when `LLM` uses `anthropic:*`. |
| `WHISPER_MODEL_SIZE` | No | `medium` | backend, worker | Whisper size: `tiny`, `base`, `small`, `medium`, `large`. |
| `TEMP_DIR` | No | `temp` (local) / `/app/uploads` (Docker) | backend, worker | Working directory for uploaded/downloaded files and clip output paths. |
| `DATABASE_URL` | Yes | compose-provided value | backend, worker | Postgres connection string. |
| `REDIS_HOST` | Yes (Docker) | `localhost` | backend, worker | Redis host. |
| `REDIS_PORT` | No | `6379` | backend, worker | Redis port. |
| `BETTER_AUTH_SECRET` | Yes for production | dev placeholder | frontend | Must be randomized for production. |
| `POSTGRES_DB` | Yes (Docker setup) | `supoclip` | postgres init | Database name for compose setup. |
| `POSTGRES_USER` | Yes (Docker setup) | `supoclip` | postgres init | Database user for compose setup. |
| `POSTGRES_PASSWORD` | Yes (Docker setup) | `supoclip_password` | postgres init | Database password for compose setup. |

## Backward Compatibility Variables

These are accepted for compatibility with older local setups:

| Legacy Variable | Preferred Variable |
|---|---|
| `LLM_MODEL` | `LLM` |
| `WHISPER_MODEL` | `WHISPER_MODEL_SIZE` |

## Model String Format

Use `provider:model`.

Examples:
- `openai:gpt-5`
- `openai:gpt-5-mini`
- `openai:gpt-4.1`
- `anthropic:claude-4-sonnet`
- `google:gemini-2.5-pro`

## Entrypoint Alignment

- Docker backend command uses `src.main_refactored:app`.
- Local development can use either `src.main_refactored:app` (recommended) or `src.main:app`.

## Validation Checklist

When adding/changing a variable:
1. Update `backend/src/config.py`.
2. Update `.env.example` and `backend/.env.example`.
3. Update this file.
4. Update references in `QUICKSTART.md` and `CLAUDE.md` if user-visible.
