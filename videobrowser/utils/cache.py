import json
import os
import shutil
from pathlib import Path
from videobrowser.config import get_config
from videobrowser.utils.parser import extract_youtube_id

class CacheManager:
    def __init__(self):
        self.config = get_config()
        # Ensure base cache directories exist
        # We can default to 'data/cache' if not specified in config
        self.base_dir = Path("data/cache")
        self.video_dir = self.base_dir / "videos"
        self.audio_dir = self.base_dir / "audio"
        self.transcript_dir = self.base_dir / "transcripts"
        self.transcript_with_timestamps_dir = self.base_dir / "transcripts_with_timestamps"
        
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_with_timestamps_dir.mkdir(parents=True, exist_ok=True)

    def _get_id(self, video_url: str) -> str:
        """Extract simplified video ID from URL"""
        return extract_youtube_id(video_url)

    def has_transcript(self, video_url: str) -> bool:
        vid = self._get_id(video_url)
        return (self.transcript_dir / f"{vid}.txt").exists()

    def get_transcript(self, video_url: str) -> str:
        vid = self._get_id(video_url)
        path = self.transcript_dir / f"{vid}.txt"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def save_transcript(self, video_url: str, text: str):
        vid = self._get_id(video_url)
        path = self.transcript_dir / f"{vid}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def has_transcript_with_timestamps(self, video_url: str) -> bool:
        vid = self._get_id(video_url)
        return (self.transcript_with_timestamps_dir / f"{vid}.json").exists()

    def get_transcript_with_timestamps(self, video_url: str) -> list[dict]:
        vid = self._get_id(video_url)
        path = self.transcript_with_timestamps_dir / f"{vid}.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load cached transcript with timestamps: {e}")
                return None
        return None

    def save_transcript_with_timestamps(self, video_url: str, segments: list[dict]):
        vid = self._get_id(video_url)
        path = self.transcript_with_timestamps_dir / f"{vid}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
            
    def has_video(self, video_url: str) -> bool:
        vid = self._get_id(video_url)
        # Check common video extensions
        for ext in [".mp4", ".webm", ".mkv"]:
             if (self.video_dir / f"{vid}{ext}").exists():
                 return True
        return False

    def get_video_path(self, video_url: str) -> str:
        vid = self._get_id(video_url)
        # Return first matching file
        for ext in [".mp4", ".webm", ".mkv"]:
             path = self.video_dir / f"{vid}{ext}"
             if path.exists():
                 return str(path)
        return None

    def get_video_storage_path(self, video_url: str, ext: str = "mp4") -> str:
        """Returns the intended path for saving a new video file"""
        vid = self._get_id(video_url)
        return str(self.video_dir / f"{vid}.{ext}")

    def has_audio(self, video_url: str) -> bool:
        vid = self._get_id(video_url)
        # Check common audio extensions
        for ext in [".mp3", ".m4a", ".wav"]:
             if (self.audio_dir / f"{vid}{ext}").exists():
                 return True
        return False

    def get_audio_path(self, video_url: str) -> str:
        vid = self._get_id(video_url)
        # Return first matching file
        for ext in [".mp3", ".m4a", ".wav"]:
             path = self.audio_dir / f"{vid}{ext}"
             if path.exists():
                 return str(path)
        return None

    def get_audio_storage_path(self, video_url: str, ext: str = "mp3") -> str:
        """Returns the intended path for saving a new audio file"""
        vid = self._get_id(video_url)
        return str(self.audio_dir / f"{vid}.{ext}")

    def save_caption(self, video_url: str, caption: str):
        vid = self._get_id(video_url)
        caption_dir = self.base_dir / "captions"
        caption_dir.mkdir(parents=True, exist_ok=True)
        path = caption_dir / f"{vid}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(caption)

    def get_caption(self, video_url: str) -> str:
        vid = self._get_id(video_url)
        caption_dir = self.base_dir / "captions"
        path = caption_dir / f"{vid}.txt"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return None

# Singleton instance for easy access
cache_manager = CacheManager()
