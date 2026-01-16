import yaml
import os
from pathlib import Path
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()

class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

class LLMSettings(BaseModel):
    default: LLMConfig
    # Use Dict[str, Any] for overrides to allow partial updates (patching)
    overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("overrides", mode="before")
    @classmethod
    def ensure_dict(cls, v):
        return v or {}

class TranscriptConfig(BaseModel):
    provider: str = "local"
    oxylabs_username: Optional[str] = Field(default_factory=lambda: os.getenv("OXYLABS_USERNAME"))
    oxylabs_password: Optional[str] = Field(default_factory=lambda: os.getenv("OXYLABS_PASSWORD"))

class WatcherConfig(BaseModel):
    num_frames: int = 10
    video_downloader: str = "ytdlp" # "ytdlp" | "pytubefix"

class SelectorConfig(BaseModel):
    top_k: int = 5

class CacheConfig(BaseModel):
    enabled: bool = True
    base_dir: str = "data/cache"

class SearchConfig(BaseModel):
    text_search_provider: Optional[str] = "tavily"  # "tavily" | "serper" | "duckduckgo" | None
    video_search_provider: str = "youtube" # "youtube" | "serper" | "duckduckgo"

class CheckerConfig(BaseModel):
    max_loop_steps: int = 3

class PlannerConfig(BaseModel):
    max_queries: int = 3

class LoggerConfig(BaseModel):
    enabled: bool = True
    log_dir: str = "data/logs"

class PromptsConfig(BaseModel):
    analyst_format_instructions: Optional[str] = None

class AppConfig(BaseModel):
    llm: LLMSettings
    transcript: TranscriptConfig = Field(default_factory=TranscriptConfig)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)
    selector: SelectorConfig = Field(default_factory=SelectorConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    checker: CheckerConfig = Field(default_factory=CheckerConfig)
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)

_config: Optional[AppConfig] = None

def load_config(config_path: str = "config.yaml") -> AppConfig:
    global _config
    
    # Look for config in project root (assuming execution from root)
    # or relative to this file's parent's parent (if running as package)
    path = Path(config_path)
    if not path.exists():
        # Try to find it relative to the module if not found in cwd
        module_path = Path(__file__).parent.parent.parent / config_path
        if module_path.exists():
            path = module_path
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path.absolute()}")

    with open(path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    _config = AppConfig(**config_data)
    return _config

def get_config() -> AppConfig:
    global _config
    if _config is None:
        return load_config()
    return _config
