from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    def __init__(self):
        # Backward compatible env handling:
        # - prefer the documented vars (LLM, WHISPER_MODEL_SIZE)
        # - still accept legacy names used in older revisions
        self.whisper_model = os.getenv("WHISPER_MODEL_SIZE") or os.getenv("WHISPER_MODEL", "medium")
        self.whisper_device = (os.getenv("WHISPER_DEVICE", "auto") or "auto").strip().lower()
        self.transcription_provider = (os.getenv("TRANSCRIPTION_PROVIDER", "local") or "local").strip().lower()
        self.llm = os.getenv("LLM") or os.getenv("LLM_MODEL") or "openai:gpt-5-mini"
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.zai_api_key = os.getenv("ZAI_API_KEY")
        self.assembly_ai_api_key = os.getenv("ASSEMBLY_AI_API_KEY")
        self.admin_api_key = os.getenv("ADMIN_API_KEY")
        self.secret_encryption_key = os.getenv("SECRET_ENCRYPTION_KEY")

        self.max_video_duration = int(os.getenv("MAX_VIDEO_DURATION", "3600"))
        self.output_dir = os.getenv("OUTPUT_DIR", "outputs")

        self.max_clips = int(os.getenv("MAX_CLIPS", "10"))
        self.clip_duration = int(os.getenv("CLIP_DURATION", "30"))  # seconds

        self.temp_dir = os.getenv("TEMP_DIR", "temp")

        # Redis configuration
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.worker_max_jobs = int(os.getenv("WORKER_MAX_JOBS", "2"))
        self.arq_queue_name = (os.getenv("ARQ_QUEUE_NAME", "arq:queue:local") or "arq:queue:local").strip()
        self.arq_local_queue_name = (os.getenv("ARQ_QUEUE_NAME_LOCAL", "arq:queue:local") or "arq:queue:local").strip()
        self.arq_assembly_queue_name = (os.getenv("ARQ_QUEUE_NAME_ASSEMBLY", "arq:queue:assembly") or "arq:queue:assembly").strip()
