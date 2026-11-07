from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gnani_api_key: str = ""
    gnani_base_url: str = "https://api.vachana.ai"
    gnani_stt_ws: str = "wss://api.vachana.ai/stt/v3/stream"
    credit_budget: float = 5000.0
    stock_voice: str = "Shivani"
    stock_tts_model: str = "timbre-v3.0"
    cors_origins: str = "*"
    evon_chat_url: str = ""
    evon_model: str = "gnani-evon-v3.3"
    evon_api_key: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
