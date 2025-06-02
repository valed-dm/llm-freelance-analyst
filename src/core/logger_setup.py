import logging
import logging.handlers
import os
from pathlib import Path
import re
import sys
from typing import Any
from typing import ClassVar
from typing import Optional
from typing import cast


PALETTE = [
    "\033[31m",  # RED
    "\033[32m",  # GREEN
    "\033[33m",  # YELLOW
    "\033[34m",  # BLUE
    "\033[35m",  # MAGENTA
    "\033[36m",  # CYAN
]

try:
    from colorama import Fore
    from colorama import Style
    from colorama import init as colorama_init

    colorama_init()

except ImportError:

    class _FallbackFore:
        RED, GREEN, YELLOW, BLUE, MAGENTA, CIAN = PALETTE
        RESET = "\033[39m"

    class _FallbackStyle:
        BRIGHT = "\033[1m"
        DIM = "\033[2m"
        RESET_ALL = "\033[0m"

    Fore = cast(Any, _FallbackFore())
    Style = cast(Any, _FallbackStyle())

    def colorama_init(
        autoreset: bool = False,
        convert: Optional[bool] = None,
        strip: Optional[bool] = None,
        wrap: bool = True,
    ) -> None:
        """Colorama-compatible no-op initializer.

        Args match colorama.init() exactly:
        - autoreset: Reset colors after each print
        - convert: Enable ANSI conversion (Windows)
        - strip: Strip ANSI sequences
        - wrap: Wrap sys.stdout/stderr
        """
        pass


# Environment config
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE_NAME = os.getenv("LOG_FILE", "app.log")
LOG_FILE_SIZE_MB = int(os.getenv("LOG_FILE_SIZE_MB", "10"))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

_logging_configured = False


class ColoredFormatter(logging.Formatter):
    COLOR_CODES: ClassVar[dict[int, str]] = {
        logging.DEBUG: Fore.YELLOW,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.BLUE,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: f"{Fore.RED}{Style.BRIGHT}",
    }
    RESET: ClassVar[str] = Style.RESET_ALL

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLOR_CODES.get(record.levelno, self.RESET)
        return f"{color}{super().format(record)}{self.RESET}"


class CleanFormatter(logging.Formatter):
    ANSI_REGEX = re.compile(r"\x1b\[[0-9;]*m")

    def format(self, record: logging.LogRecord) -> str:
        return self.ANSI_REGEX.sub("", super().format(record))


def configure_logging(
    log_level: str = LOG_LEVEL,
    log_dir: str = LOG_DIR,
    log_file_name: str = LOG_FILE_NAME,
    log_file_size_mb: int = LOG_FILE_SIZE_MB,
    log_backup_count: int = LOG_BACKUP_COUNT,
) -> None:
    """
    Set up logging configuration with colorized console and rotating file handlers.
    """
    global _logging_configured

    if _logging_configured:
        return

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    log_path = Path(log_dir) / log_file_name
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Format strings
    log_format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(log_format, datefmt=date_format))

    # File
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=log_file_size_mb * 1024 * 1024,
        backupCount=log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(CleanFormatter(log_format, datefmt=date_format))

    # Apply to root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _logging_configured = True
