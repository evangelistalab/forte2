import logging
import sys

from forte2.lib import cpp_helpers

VERBOSITY_QUIET = 0
VERBOSITY_WARNING = 1
VERBOSITY_ESSENTIAL = 2
VERBOSITY_INFO1 = 3
VERBOSITY_INFO2 = 4
VERBOSITY_DEBUG = 5


LOGGING_LEVEL = {
    VERBOSITY_QUIET: logging.CRITICAL + 1,  # Quiet
    VERBOSITY_WARNING: logging.CRITICAL,  # Warning
    VERBOSITY_ESSENTIAL: logging.WARNING,  # Essential
    VERBOSITY_INFO1: logging.INFO,  # Info1
    VERBOSITY_INFO2: logging.INFO - 1,  # Info2
    VERBOSITY_DEBUG: logging.DEBUG,
}


class LoggerConfig:
    """Centralized logging configuration that matches C++ behavior"""

    _initialized = False
    _verbosity_level = 3

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

        cls._verbosity_level = level


# Global convenience functions
def set_verbosity_level(level):
    LoggerConfig.set_log_level(level)
    cpp_helpers.set_log_level(level)  # Ensure the C++ side also uses the same level


def get_verbosity_level():
    """Get the current verbosity level"""
    return LoggerConfig._verbosity_level


def log(message, level=2):
    level = max(level, VERBOSITY_WARNING)
    logging.log(LOGGING_LEVEL.get(level, logging.INFO), message)


def log_warning(message):
    logging.critical(message)


def log_essential(message):
    logging.warning(message)


def log_info1(message):
    logging.info(message)


def log_info2(message):
    logging.log(logging.INFO - 1, message)


def log_debug(message):
    logging.debug(message)


# Auto-setup on import
LoggerConfig.setup()
