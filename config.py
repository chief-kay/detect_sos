"""
config.py
Purpose: Centralized configuration file for UAV Surveillance Drone System
"""

import numpy as np
import cv2

CONFIG = {
    # === Camera Settings ===
    "CAMERA_FPS": 30,
    "CAMERA_RESOLUTION": (640, 480),
    "CAMERA_INDEX": 0,
    "CAMERA_WIDTH": 640,
    "CAMERA_HEIGHT": 480,
    # === Speed Estimation ===
    "PIXEL_TO_METER_RATIO": 0.05,  # 1 pixel = 0.05 meters (adjust based on calibration)
    "SPEED_LIMIT_KMPH": 80,  # km/h for overspeed alert

    # === Gesture Recognition ===
    "GESTURE_CONFIDENCE_THRESHOLD": 0.8,

    # === Logging ===
    "LOG_CSV_PATH": "logs/events.csv",
    "LOG_JSON_PATH": "logs/events.json",
    "ENABLE_DEBUG_LOGGING": True,

    # === Navigation ===
    "GPS_TIMEOUT": 15,  # seconds before declaring GPS failure
    "FAILSAFE_ALTITUDE": 10,  # meters to maintain during autonomous mode fallback

    # === Diagnostic Thresholds ===
    "BATTERY_MIN_LEVEL": 40,  # Minimum % to allow takeoff
    "CPU_MAX_USAGE": 85,      # %
    "MEMORY_MAX_USAGE": 85,   # %
    "SIGNAL_MIN_RSSI": 10,    # RSSI threshold

    # === Manual Control ===
    "MANUAL_OVERRIDE_TIMEOUT": 10,  # seconds of inactivity before reverting to autonomous

    # === Alert System ===
    "ENABLE_AUDIO_ALERTS": True,
    "ALERT_VOLUME": 0.7,

    # === Return-to-Base ===
    "BATTERY_CRITICAL_THRESHOLD": 25,  # % battery to trigger RTB

    # === Test Runner Settings ===
    "TEST_RETRIES": 3,
    "TEST_DELAY_BETWEEN": 1.0,  # seconds

    # === Reserved for future features ===
    "AI_MODEL_PATH": "models/detector_v2.tflite",
    "PLATE_DB_PATH": "data/plates.db",
}

class Config:
    """Configuration class with methods for accessing settings and test utilities."""
    
    def __init__(self):
        self._config = CONFIG
    
    def get(self, key, default=None):
        """Get a configuration value by key."""
        return self._config.get(key, default)
    
    def set(self, key, value):
        """Set a configuration value."""
        self._config[key] = value
    
    def get_camera_settings(self):
        """Get camera-related settings as a dictionary."""
        return {
            "fps": self.get("CAMERA_FPS"),
            "resolution": self.get("CAMERA_RESOLUTION"),
            "index": self.get("CAMERA_INDEX"),
            "width": self.get("CAMERA_WIDTH"),
            "height": self.get("CAMERA_HEIGHT")
        }
    
    def get_test_settings(self):
        """Get test runner settings."""
        return {
            "retries": self.get("TEST_RETRIES"),
            "delay": self.get("TEST_DELAY_BETWEEN")
        }
    
    def get_dummy_frame(self):
        """Generate a dummy frame for testing purposes."""
        width = self.get("CAMERA_WIDTH", 640)
        height = self.get("CAMERA_HEIGHT", 480)
        
        # Create a simple test pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)  # type: ignore
        
        # Add some patterns for testing
        cv2.rectangle(frame, (50, 50), (150, 150), (0, 255, 0), 2)  # Green rectangle
        cv2.circle(frame, (300, 200), 50, (255, 0, 0), -1)  # Blue circle
        cv2.putText(frame, "TEST", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        
        return frame
    
    def validate_config(self):
        """Validate that all required configuration keys exist."""
        required_keys = [
            "CAMERA_FPS", "CAMERA_RESOLUTION", "CAMERA_INDEX",
            "PIXEL_TO_METER_RATIO", "SPEED_LIMIT_KMPH",
            "GESTURE_CONFIDENCE_THRESHOLD", "LOG_CSV_PATH"
        ]
        
        missing_keys = []
        for key in required_keys:
            if key not in self._config:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        return True
    
    def get_all_config(self):
        """Get the entire configuration dictionary."""
        return self._config.copy()
    
    def is_debug_enabled(self):
        """Check if debug logging is enabled."""
        return self.get("ENABLE_DEBUG_LOGGING", False)
    
    def get_thresholds(self):
        """Get all threshold values for diagnostics."""
        return {
            "battery_min": self.get("BATTERY_MIN_LEVEL"),
            "cpu_max": self.get("CPU_MAX_USAGE"),
            "memory_max": self.get("MEMORY_MAX_USAGE"),
            "signal_min": self.get("SIGNAL_MIN_RSSI"),
            "battery_critical": self.get("BATTERY_CRITICAL_THRESHOLD")
        }