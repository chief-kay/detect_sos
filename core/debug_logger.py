"""
debug_logger.py
Purpose: Unified logging system for UAV diagnostics and mission monitoring.
"""

import logging
import os
from datetime import datetime

class DebugLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"uav_log_{timestamp}.log")

        # Logger instance
        self.logger = logging.getLogger("UAVLogger")
        self.logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers during repeated instantiations
        if not self.logger.handlers:
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(message)s",
                datefmt='%Y-%m-%d %H:%M:%S'
            ))

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def exception(self, message):
        self.logger.exception(message)
