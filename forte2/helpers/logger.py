import logging
import sys
import forte2


class LoggerConfig:
    """Centralized logging configuration that matches C++ behavior"""

    _initialized = False

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
        root_logger.setLevel(logging.INFO)

        cls._initialized = True

    @staticmethod
    def set_log_level(level):
        """Set log level using same numbering as C++ (0-4)"""

        level_map = {
            0: logging.CRITICAL + 1,  # No output
            1: logging.ERROR,
            2: logging.WARNING,
            3: logging.INFO,
            4: logging.DEBUG,
        }

        if level in level_map:
            logging.getLogger().setLevel(level_map[level])
        else:
            raise ValueError(f"Invalid log level: {level}")


# Global convenience functions
def set_verbosity_level(level):
    LoggerConfig.set_log_level(level)
    forte2.set_log_level(level)  # Ensure the C++ side also uses the same level


# Auto-setup on import
LoggerConfig.setup()
