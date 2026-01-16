import json
import time
import os
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path
from videobrowser.config import get_config

class TraceLogger:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TraceLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, log_dir: str = "data/logs", enabled: bool = True):
        if self._initialized:
            return
            
        self.enabled = enabled
        if not self.enabled:
            self._initialized = True
            return

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a new log file for each session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"trace_{timestamp}.jsonl"
        self._initialized = True
        
        print(f"📄 Logging execution trace to: {self.log_file}")

    def log(self, step: str, action: str, details: Optional[Dict[str, Any]] = None, level: str = "INFO"):
        """
        Log a single event to the JSONL file.
        
        Args:
            step: The name of the node or execution step (e.g., "Planner", "Searcher").
            action: A short description of the action (e.g., "generated_plan", "fetched_results").
            details: A dictionary containing relevant data (queries, results, reasoning, etc.).
            level: Log level (INFO, WARN, ERROR).
        """
        if not self.enabled:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "step": step,
            "action": action,
            "details": details or {}
        }
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠️ Failed to write log: {e}")

# Global instance accessor
_logger = None

def get_logger() -> TraceLogger:
    global _logger
    if _logger is None:
        try:
            config = get_config()
            log_dir = config.logger.log_dir
            enabled = config.logger.enabled
        except Exception:
            # Fallback if config isn't loaded yet or fails
            log_dir = "data/logs"
            enabled = True
            
        _logger = TraceLogger(log_dir=log_dir, enabled=enabled)
    return _logger
