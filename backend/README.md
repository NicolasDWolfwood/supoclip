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
- root `.env.sample`
- `docs/config.md`
- `docs/local-host-mappings.md`

Backend compatibility behavior:
- preferred model var: `LLM`
- legacy model var: `LLM_MODEL`
- AI provider keys: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `ZAI_API_KEY`
- z.ai requests use Coding API endpoint: `https://api.z.ai/api/coding/paas/v4`
- z.ai supports user key profiles in Settings (`subscription`, `metered`) with routing mode (`auto`, `subscription`, `metered`)
- preferred Whisper var: `WHISPER_MODEL_SIZE`
- legacy Whisper var: `WHISPER_MODEL`
- transcription provider var: `TRANSCRIPTION_PROVIDER` (`local` default, `assemblyai` optional)

## Docker

Backend is started by `docker-compose.yml` with:
- app entrypoint: `src.main_refactored:app`
- worker: `src.workers.tasks.WorkerSettings`

## API Docs

When running locally or via Docker:
- Swagger UI: `http://${APP_HOST}:${BACKEND_HOST_PORT}/docs` (default `http://localhost:8000/docs`)
- AI model discovery: `GET /tasks/ai-settings/{provider}/models` (uses saved user key first, then env fallback)
- z.ai profile keys:
  - `PUT /tasks/ai-settings/zai/profiles/{subscription|metered}/key`
  - `DELETE /tasks/ai-settings/zai/profiles/{subscription|metered}/key`
  - `PUT /tasks/ai-settings/zai/routing-mode` (`auto`, `subscription`, `metered`)

## Fonts

- `GET /fonts`: list available subtitle fonts from `backend/fonts`.
- `POST /fonts/upload`: upload a new `.ttf` font file (available immediately in the frontend font dropdown).
- `GET /fonts/{font_name}`: serve a specific `.ttf` font for frontend preview loading.
- You can still add `.ttf` files manually to `backend/fonts/` if preferred.
- Install a curated top-10 free subtitle pack: `./backend/bin/install_subtitle_font_pack.sh`
- Pack details and references: `backend/fonts/FONT_PACK_TOP10.md`

## Admin Task API

- `POST /tasks/admin/cancel-all`: cancel all `queued`/`processing` tasks and drain queued ARQ jobs.
- If `ADMIN_API_KEY` is set, include it as `x-admin-key` request header.
