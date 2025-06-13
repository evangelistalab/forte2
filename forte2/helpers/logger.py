import logging
import sys
import forte2

LOGGING_LEVEL = {
    0: logging.CRITICAL + 1,  # Quiet
    1: logging.CRITICAL,  # Warning
    2: logging.WARNING,  # Essential
    3: logging.INFO,  # Info1
    4: logging.INFO + 1,  # Info2
    5: logging.DEBUG,
}


class LoggerConfig:
    """Centralized logging configuration that matches C++ behavior"""

    _initialized = False
    _verbosity_level = 2

    @classmethod
    def setup(cls):
        """Initialize logging configuration once"""
        if cls._initialized:
            return

        # Create formatter
        formatter = logging.Formatter("%(message)s")

        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(console_handler)
        # default to INFO level, like on the C++ side
        root_logger.setLevel(LOGGING_LEVEL[cls._verbosity_level])

        cls._initialized = True

    @classmethod
    def set_log_level(cls, level):
        """Set log level using same numbering as C++ (0-4)"""

        if level in LOGGING_LEVEL:
            logging.getLogger().setLevel(LOGGING_LEVEL[level])
        else:
            raise ValueError(f"Invalid log level: {level}")

        cls.verbosity_level = level


# Global convenience functions
def set_verbosity_level(level):
    LoggerConfig.set_log_level(level)
    forte2.set_log_level(level)  # Ensure the C++ side also uses the same level


def get_verbosity_level():
    """Get the current verbosity level"""
    return LoggerConfig._verbosity_level


def log(message, level=2):
    logging.log(LOGGING_LEVEL.get(level, logging.INFO), message)


def log_warning(message):
    logging.critical(message)


def log_essential(message):
    logging.warning(message)


def log_info1(message):
    logging.info(message)


def log_info2(message):
    logging.log(logging.INFO + 1, message)


def log_debug(message):
    logging.debug(message)


# Auto-setup on import
LoggerConfig.setup()
