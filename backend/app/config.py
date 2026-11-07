from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLMT_", env_file=ROOT / ".env", extra="ignore")

    data_dir: Path = ROOT / "data"
    host: str = "127.0.0.1"
    port: int = 8000
    default_embed_model: str = "intfloat/multilingual-e5-small"
    # Path to a llama.cpp checkout (to convert to GGUF). Defaults to tools/llama.cpp if it exists.
    llama_cpp_dir: Path | None = None
    hf_token: str | None = None
    # Optional key for the OpenAI-compatible API (/v1). Empty = no authentication.
    api_key: str | None = None
    # Automatic backups of the database, datasets and RAG
    backup_daily: bool = True
    backup_keep: int = 7

    @property
    def llama_cpp(self) -> Path | None:
        d = self.llama_cpp_dir or ROOT / "tools" / "llama.cpp"
        return d if (d / "convert_hf_to_gguf.py").exists() else None

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def rag_dir(self) -> Path:
        return self.data_dir / "rag"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "llmtrainer.db"


settings = Settings()
for d in (settings.datasets_dir, settings.models_dir, settings.outputs_dir, settings.jobs_dir, settings.rag_dir):
    d.mkdir(parents=True, exist_ok=True)
