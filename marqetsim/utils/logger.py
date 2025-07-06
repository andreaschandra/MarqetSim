import re
from logging import Formatter, Logger, StreamHandler, handlers
from pathlib import Path
from colorama import Fore, Style


class ColorFormatter(Formatter):

    COLORS = {
        "DEBUG": Fore.LIGHTCYAN_EX,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA + Style.BRIGHT,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"

        return super().format(record)


class RecordFormater(Formatter):
    def format(self, record):
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

        record.msg = ansi_escape.sub("", record.msg)
        return super().format(record)


class LogCreator(Logger):
    def __init__(self, name: str, level="DEBUG", log_file: Path = None):
        super().__init__(name, level)

        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Console handler with color formatting
        if not any(isinstance(handler, StreamHandler) for handler in self.handlers):
            console_handler = StreamHandler()
            console_handler.setFormatter(ColorFormatter(log_format))
            self.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            if not any(
                isinstance(handler, handlers.RotatingFileHandler)
                for handler in self.handlers
            ):
                file_handler = handlers.RotatingFileHandler(
                    filename=log_path,
                    maxBytes=10 * 1024 * 1024,
                    backupCount=1,
                    encoding="utf-8",
                )
                file_handler.setFormatter(RecordFormater(log_format))
                self.addHandler(file_handler)


if __name__ == "__main__":
    logger = LogCreator("test_logger", level="DEBUG")

    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.debug("This is a debug message.")
    logger.critical("This is a critical message.")
