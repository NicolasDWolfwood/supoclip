# Backend

FastAPI service responsible for:
- source ingestion (YouTube/upload)
- transcription + transcript analysis
- clip rendering and storage
- task progress/status APIs

## Prerequisites

- Python 3.11+
- `uv`
- `ffmpeg`

Install ffmpeg:

```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt update -y && sudo apt install ffmpeg -y

# Windows (Chocolatey)
choco install ffmpeg
```

## Local Development

```bash
cd backend
uv venv .venv
source .venv/bin/activate
uv sync
uvicorn src.main_refactored:app --reload --host 0.0.0.0 --port 8000
```

Legacy entrypoint (still present):

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## Configuration

Use the root env file and canonical config reference:
- `.env.example`
- `docs/config.md`

Backend compatibility behavior:
- preferred model var: `LLM`
- legacy model var: `LLM_MODEL`
- preferred Whisper var: `WHISPER_MODEL_SIZE`
- legacy Whisper var: `WHISPER_MODEL`
- transcription provider var: `TRANSCRIPTION_PROVIDER` (`local` default, `assemblyai` optional)

## Docker

Backend is started by `docker-compose.yml` with:
- app entrypoint: `src.main_refactored:app`
- worker: `src.workers.tasks.WorkerSettings`

## API Docs

When running locally or via Docker:
- Swagger UI: http://localhost:8000/docs

## Admin Task API

- `POST /tasks/admin/cancel-all`: cancel all `queued`/`processing` tasks and drain queued ARQ jobs.
- If `ADMIN_API_KEY` is set, include it as `x-admin-key` request header.
