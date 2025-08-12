"""
camera_stream.py
Purpose: Threaded video stream capture for real-time UAV vision processing.
"""

import cv2
import threading
import time
import logging
from typing import Optional, Tuple
import numpy as np

class CameraStream:
    def __init__(self, src=0, width=640, height=480, fps=30):
        """
        Initialize camera stream.
        
        Args:
            src: Camera source (0 for default camera, or video file path)
            width: Frame width
            height: Frame height
            fps: Target frames per second
        """
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_delay = 1.0 / fps
        
        # Camera setup
        self.capture = None
        self.frame = None
        self.running = False
        self.lock = threading.RLock()  # RLock allows multiple acquires by same thread
        self.thread = None
        
        # Statistics
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_frame_time = 0
        
        # Error handling
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Initialize camera
        self._initialize_camera()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _initialize_camera(self):
        """Initialize camera capture with error handling."""
        try:
            self.capture = cv2.VideoCapture(self.src)
            
            if not self.capture.isOpened():
                raise IOError(f"[CAMERA] Unable to open video source: {self.src}")
            
            # Set camera properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Set buffer size to reduce latency
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verify settings
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
            
            if hasattr(self, 'logger'):
                self.logger.info(f"[CAMERA] Initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"[CAMERA] Initialization failed: {e}")
            raise

    def start(self):
        """Start the camera stream thread."""
        if not self.capture or not self.capture.isOpened():
            self._initialize_camera()
            
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            
            # Wait for first frame
            timeout = 5.0  # 5 second timeout
            start_time = time.time()
            while self.frame is None and (time.time() - start_time) < timeout:
                time.sleep(0.1)
                
            if self.frame is None:
                self.stop()
                raise TimeoutError("[CAMERA] Failed to capture first frame within timeout")
                
            self.logger.info("[CAMERA] Stream started successfully")
        return self

    def _update_loop(self):
        """Main camera update loop with error handling."""
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            try:
                ret, frame = self.capture.read() # type: ignore
                current_time = time.time()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    self.dropped_frames += 1
                    
                    if consecutive_failures >= max_failures:
                        self.logger.error("[CAMERA] Too many consecutive frame failures")
                        break
                        
                    time.sleep(self.retry_delay)
                    continue
                
                # Reset failure counter on success
                consecutive_failures = 0
                
                # Update frame with thread safety
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
                    self.last_frame_time = current_time
                
                # Maintain target FPS
                time.sleep(self.frame_delay)
                
            except Exception as e:
                consecutive_failures += 1
                self.logger.error(f"[CAMERA] Update loop error: {e}")
                if consecutive_failures >= max_failures:
                    break
                time.sleep(self.retry_delay)
        
        self.logger.warning("[CAMERA] Update loop ended")

    def read(self) -> Optional[np.ndarray]:
        """
        Read the latest frame.
        
        Returns:
            Latest frame as numpy array, or None if no frame available
        """
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def read_with_timestamp(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Read the latest frame with timestamp.
        
        Returns:
            Tuple of (frame, timestamp) or (None, 0) if no frame available
        """
        with self.lock:
            if self.frame is not None:
                return self.frame.copy(), self.last_frame_time
            return None, 0

    def stop(self):
        """Stop the camera stream and cleanup resources."""
        self.running = False
        
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                self.logger.warning("[CAMERA] Thread did not stop gracefully")
        
        if self.capture is not None:
            self.capture.release()
            
        cv2.destroyAllWindows()
        self.logger.info("[CAMERA] Stream stopped")

    def is_active(self) -> bool:
        """Check if camera stream is active and healthy."""
        if not self.running or not self.capture:
            return False
            
        if not self.capture.isOpened():
            return False
            
        # Check if we're receiving recent frames
        if self.last_frame_time > 0:
            time_since_last_frame = time.time() - self.last_frame_time
            if time_since_last_frame > 2.0:  # No frame for 2 seconds
                return False
                
        return True

    def get_stats(self) -> dict:
        """Get camera stream statistics."""
        return {
            "frame_count": self.frame_count,
            "dropped_frames": self.dropped_frames,
            "fps": self.fps,
            "resolution": (self.width, self.height),
            "last_frame_time": self.last_frame_time,
            "is_active": self.is_active()
        }

    def restart(self):
        """Restart the camera stream."""
        self.logger.info("[CAMERA] Restarting stream...")
        self.stop()
        time.sleep(1.0)
        self._initialize_camera()
        return self.start()

    def __enter__(self):
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

# Standalone test with improved error handling
if __name__ == "__main__":
    import argparse
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Test camera stream")
    parser.add_argument("--source", type=int, default=0, help="Camera source index")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    args = parser.parse_args()
    
    try:
        # Use context manager for automatic cleanup
        with CameraStream(src=args.source, width=args.width, height=args.height, fps=args.fps) as cam:
            print("[INFO] Camera started. Press 'q' to quit, 's' to show stats")
            
            while True:
                frame = cam.read()
                if frame is not None:
                    cv2.imshow("Camera Preview", frame)
                else:
                    print("[WARNING] No frame received")
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    stats = cam.get_stats()
                    print(f"[STATS] {stats}")
                    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}")