import logging
import logging.config
import datetime
import os
from typing import Dict, Any
from pathlib import Path
from config import settings


def setup_logging() -> None:
    """Setup logging configuration"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = logs_dir / f"bodh.ai_{current_time}.log"

    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "verbose": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "simple": {"format": "%(levelname)s %(message)s"},
        },
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "simple",
            },
            "file": {
                "level": "DEBUG",
                "class": "logging.FileHandler",
                "filename": log_file,
                "formatter": "verbose",
            },
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": "DEBUG" if settings.DEBUG_MODE else "INFO",
                "propagate": True,
            },
            "uvicorn": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(logging_config)
    logging.info("Logging configured successfully")
