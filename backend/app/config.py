from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gnani_api_key: str = ""
    gnani_base_url: str = "https://api.vachana.ai"
    gnani_stt_ws: str = "wss://api.vachana.ai/stt/v3/stream"
    credit_budget: float = 5000.0
    cors_origins: str = "*"
    llm_provider: str = "canned"
    llm_api_key: str = ""
    llm_chat_url: str = ""
    llm_model: str = ""

    # Voice cloning. Gnani recommends 5-30 seconds of clean reference audio.
    min_enrol_seconds: float = 5.0
    max_enrol_seconds: float = 30.0
    max_upload_mb: float = 10.0
    # vachana-vc-v1 supports up to 44100 Hz; higher output sounds cleaner.
    clone_tts_sample_rate: int = 44100
    stock_tts_sample_rate: int = 48000

    # Live Prisma v2.5 streaming. v2.5 needs one concrete locale; en-IN also
    # accepts Hindi-English mixed audio and returns Latin script (Hinglish).
    stt_language: str = "en-IN"
    stt_format: str = "verbatim"
    stt_vad_threshold: float = 0.50
    stt_min_speech_ms: int = 80
    stt_min_segment_ms: int = 120
    stt_min_silence_ms: int = 700
    stt_max_speech_s: float = 30.0
    stt_idle_timeout_sec: int = 300

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
