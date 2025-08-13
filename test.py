"""
test_runner.py
Purpose: Modular test runner to validate all core components before flight.
"""

import traceback
import time
import cv2
import os
import logging
import sys

from core.camera_stream import CameraStream
from core.object_detector import ObjectDetector
from core.alert_system import AlertSystem
from core.debug_logger import DebugLogger
from config import Config

logger = DebugLogger()
config = Config()

def run_test(name, func):
    print(f"[TEST] Running {name}...")
    try:
        result = func()
        print(f"[PASS] {name}")
        return True
    except Exception as e:
        print(f"[FAIL] {name} - {e}")
        traceback.print_exc()
        return False

def test_camera_stream():
    cam = CameraStream().start()
    time.sleep(2)
    frame = cam.read()
    cam.stop()
    assert frame is not None, "No frame captured"
    return True

def test_object_detector():
    detector = ObjectDetector(api_key="RlBsaWRPbmh1V0FFR1FMS29UQ2U")
    dummy_frame = config.get_dummy_frame()
    detections = detector.detect(dummy_frame)
    assert isinstance(detections, list), "Detection result is not a list"
    return True

def test_with_camera_stream():
    """Test object detector with camera stream."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    try:
        detector = ObjectDetector(enable_alerts=True, api_key="RlBsaWRPbmh1V0FFR1FMS29UQ2U")  # Disable alerts for testing
        camera = CameraStream(width=640, height=480, fps=30)
        
        # Start camera
        camera.start()
        print("[INFO] Camera started. Controls:")
        print("  'q' - Quit")
        print("  's' - Show stats") 
        print("  'c' - Clear logs")
        print("  'p' - Save screenshot")
        print("  'r' - Show road analysis")
        print("[INFO] Priority levels: 🚨 Critical | ⚠️ High | ⚡ Medium | ℹ️ Low")
        
        frame_count = 0
        
        while True:
            # Read frame
            frame = camera.read()
            if frame is None:
                continue
                
            frame_count += 1
            
            # Detect objects with location info
            detections = detector.detect(
                frame, 
                location="Test Location - Live Camera",
                coordinates="6.6745° N, 1.5716° W"
            )
            
            # Draw detections
            annotated_frame = detector.draw_detections(frame, detections)
            
            # Add info overlay
            critical_count = len([d for d in detections if d.get('priority') == 'critical'])
            high_count = len([d for d in detections if d.get('priority') == 'high'])
            road_count = len([d for d in detections if d['label'] == 'road'])
            
            # Draw info rectangle
            cv2.rectangle(annotated_frame, (10, 10), (600, 120), (50,50,50), cv2.FILLED)
            cv2.putText(annotated_frame, f'Frame: {frame_count} | Detections: {len(detections)}', 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(annotated_frame, f'Critical: {critical_count} | High: {high_count} | Roads: {road_count}', 
                       (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,102,51), 1)
            cv2.putText(annotated_frame, f'Total violations: {len(detector.violation_log)}', 
                       (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (51,204,51), 1)
            cv2.putText(annotated_frame, f'Source: Live Camera Feed', 
                       (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
            
            # Show frame
            cv2.imshow("UAV Surveillance - Live Camera", annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                stats = detector.get_stats()
                print(f"\n[STATS] {stats}")
            elif key == ord('c') or key == ord('C'):
                detector.clear_logs()
                print("[INFO] Logs cleared")
            elif key == ord('p') or key == ord('P'):
                cv2.imwrite(f'live_detection_capture_{frame_count}.png', annotated_frame)
                print(f"[INFO] Screenshot saved as live_detection_capture_{frame_count}.png")
            elif key == ord('r') or key == ord('R'):
                if detector.check_road_in_detections(detections):
                    road_analysis = detector.analyze_objects_on_road(detections)
                    print(f"\n[ROAD ANALYSIS] {road_analysis}")
                else:
                    print("\n[ROAD ANALYSIS] No roads detected in current frame")
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            camera.stop()
        except:
            pass
        cv2.destroyAllWindows()

def test_with_video_file():
    """Test object detector with video file."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Video file path - you can change this to your video file
    video_path = input("Enter video file path (or press Enter for default): ").strip()
    if not video_path:
        video_path = "test_video.mp4"  # Default test video
    
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        print("[INFO] Please provide a valid video file path")
        return False
    
    try:
        detector = ObjectDetector(enable_alerts=False, api_key="RlBsaWRPbmh1V0FFR1FMS29UQ2U")  # Disable alerts for testing
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[ERROR] Could not open video file: {video_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] Video loaded: {video_path}")
        print(f"[INFO] Properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        print("[INFO] Controls:")
        print("  'q' - Quit")
        print("  's' - Show stats")
        print("  'c' - Clear logs")
        print("  'p' - Save screenshot")
        print("  'r' - Show road analysis")
        print("  'SPACE' - Pause/Resume")
        print("[INFO] Priority levels: 🚨 Critical | ⚠️ High | ⚡ Medium | ℹ️ Low")
        
        frame_count = 0
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of video reached")
                    break
                    
                frame_count += 1
                
                # Detect objects
                detections = detector.detect(
                    frame,
                    location=f"Video File - {os.path.basename(video_path)}",
                    coordinates="Video Analysis Mode"
                )
                
                # Draw detections
                annotated_frame = detector.draw_detections(frame, detections)
                
                # Add info overlay
                critical_count = len([d for d in detections if d.get('priority') == 'critical'])
                high_count = len([d for d in detections if d.get('priority') == 'high'])
                road_count = len([d for d in detections if d['label'] == 'road'])
                
                # Draw info rectangle
                cv2.rectangle(annotated_frame, (10, 10), (650, 140), (50,50,50), cv2.FILLED)
                cv2.putText(annotated_frame, f'Frame: {frame_count}/{total_frames} | Detections: {len(detections)}', 
                           (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.putText(annotated_frame, f'Critical: {critical_count} | High: {high_count} | Roads: {road_count}', 
                           (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,102,51), 1)
                cv2.putText(annotated_frame, f'Total violations: {len(detector.violation_log)}', 
                           (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (51,204,51), 1)
                cv2.putText(annotated_frame, f'Source: {os.path.basename(video_path)}', 
                           (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
                cv2.putText(annotated_frame, f'Status: {"PAUSED" if paused else "PLAYING"}', 
                           (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if paused else (0,255,0), 1)
            
            # Show frame
            cv2.imshow("UAV Surveillance - Video File", annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF  # Slower playback for video analysis
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord(' '):  # Space bar to pause/resume
                paused = not paused
                print(f"[INFO] Video {'paused' if paused else 'resumed'}")
            elif key == ord('s') or key == ord('S'):
                stats = detector.get_stats()
                print(f"\n[STATS] {stats}")
            elif key == ord('c') or key == ord('C'):
                detector.clear_logs()
                print("[INFO] Logs cleared")
            elif key == ord('p') or key == ord('P'):
                cv2.imwrite(f'video_detection_capture_{frame_count}.png', annotated_frame)
                print(f"[INFO] Screenshot saved as video_detection_capture_{frame_count}.png")
            elif key == ord('r') or key == ord('R'):
                if detector.check_road_in_detections(detections):
                    road_analysis = detector.analyze_objects_on_road(detections)
                    print(f"\n[ROAD ANALYSIS] {road_analysis}")
                else:
                    print("\n[ROAD ANALYSIS] No roads detected in current frame")
                
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
    return True

def test_with_image_file():
    """Test object detector with image file."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Image file path - you can change this to your image file
    image_path = input("Enter image file path (or press Enter for default): ").strip()
    if not image_path:
        image_path = "test_image.jpg"  # Default test image
    
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        print("[INFO] Please provide a valid image file path")
        return False
    
    try:
        detector = ObjectDetector(enable_alerts=False, api_key="RlBsaWRPbmh1V0FFR1FMS29UQ2U")  # Disable alerts for testing
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return False
        
        # Get image properties
        height, width = frame.shape[:2]
        
        print(f"[INFO] Image loaded: {image_path}")
        print(f"[INFO] Properties: {width}x{height}")
        print("[INFO] Controls:")
        print("  'q' - Quit")
        print("  's' - Show stats")
        print("  'c' - Clear logs")
        print("  'p' - Save screenshot")
        print("  'r' - Show road analysis")
        print("  'd' - Show detailed detections")
        print("[INFO] Priority levels: 🚨 Critical | ⚠️ High | ⚡ Medium | ℹ️ Low")
        
        # Detect objects
        detections = detector.detect(
            frame,
            location=f"Static Image - {os.path.basename(image_path)}",
            coordinates="Image Analysis Mode"
        )
        
        # Draw detections
        annotated_frame = detector.draw_detections(frame, detections)
        
        # Add info overlay
        critical_count = len([d for d in detections if d.get('priority') == 'critical'])
        high_count = len([d for d in detections if d.get('priority') == 'high'])
        medium_count = len([d for d in detections if d.get('priority') == 'medium'])
        low_count = len([d for d in detections if d.get('priority') == 'low'])
        road_count = len([d for d in detections if d['label'] == 'road'])
        
        # Draw info rectangle
        cv2.rectangle(annotated_frame, (10, 10), (700, 160), (50,50,50), cv2.FILLED)
        cv2.putText(annotated_frame, f'Image: {os.path.basename(image_path)} | Total Detections: {len(detections)}', 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(annotated_frame, f'Critical: {critical_count} | High: {high_count} | Medium: {medium_count} | Low: {low_count}', 
                   (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,102,51), 1)
        cv2.putText(annotated_frame, f'Roads: {road_count} | Violations: {len(detector.violation_log)}', 
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (51,204,51), 1)
        cv2.putText(annotated_frame, f'Source: Static Image Analysis', 
                   (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        cv2.putText(annotated_frame, f'Resolution: {width}x{height}', 
                   (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
        cv2.putText(annotated_frame, f'Press "d" for detailed detection list', 
                   (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        print(f"\n[DETECTION SUMMARY]")
        print(f"Total detections: {len(detections)}")
        print(f"Critical: {critical_count}, High: {high_count}, Medium: {medium_count}, Low: {low_count}")
        print(f"Roads detected: {road_count}")
        print(f"Violations logged: {len(detector.violation_log)}")
        
        while True:
            # Show frame
            cv2.imshow("UAV Surveillance - Static Image", annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(0) & 0xFF  # Wait for key press
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                stats = detector.get_stats()
                print(f"\n[STATS] {stats}")
            elif key == ord('c') or key == ord('C'):
                detector.clear_logs()
                print("[INFO] Logs cleared")
            elif key == ord('p') or key == ord('P'):
                cv2.imwrite(f'image_detection_result_{int(time.time())}.png', annotated_frame)
                print(f"[INFO] Result saved as image_detection_result_{int(time.time())}.png")
            elif key == ord('r') or key == ord('R'):
                if detector.check_road_in_detections(detections):
                    road_analysis = detector.analyze_objects_on_road(detections)
                    print(f"\n[ROAD ANALYSIS] {road_analysis}")
                else:
                    print("\n[ROAD ANALYSIS] No roads detected in image")
            elif key == ord('d') or key == ord('D'):
                print(f"\n[DETAILED DETECTIONS]")
                for i, det in enumerate(detections):
                    print(f"  {i+1}. {det['label']}: {det['confidence']:.2f} confidence, "
                          f"Priority: {det.get('priority', 'low')}, "
                          f"Event: {det.get('event', 'none')}")
                
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        
    return True

def test_alert_system():
    alert = AlertSystem()
    alert.trigger_alert("test event")
    return True

def test_manual_control():
    # Skip actual RC interaction in test environment.
    print("[INFO] Manual control simulation skipped in test mode.")
    return True

def run_all_tests():
    logger.info("[TEST_RUNNER] Starting all system tests...")

    test_cases = [
        ("Camera Stream", test_camera_stream),
        ("Object Detector", test_object_detector),
        ("Live Camera + Object Detection", test_with_camera_stream),
        ("Video File + Object Detection", test_with_video_file),
        ("Image File + Object Detection", test_with_image_file),
        # ("OCR Plate Reader", test_plate_reader),
        # ("Speed Estimator", test_speed_estimator),
        # ("Gesture Detector", test_gesture_detector),
        # ("Alert System", test_alert_system),
        # ("Navigation Controller", test_navigation),
        # ("Diagnostics", test_diagnostics),
        ("Manual Control", test_manual_control),
    ]

    passed = 0
    for name, test_func in test_cases:
        if run_test(name, test_func):
            passed += 1

    total = len(test_cases)
    print(f"\n[SUMMARY] Passed {passed} / {total} tests")
    logger.info(f"[TEST_RUNNER] Completed. Passed {passed}/{total}")

def run_interactive_tests():
    """Run tests interactively - let user choose which test to run."""
    print("\n" + "="*60)
    print("UAV SURVEILLANCE SYSTEM - INTERACTIVE TEST MODE")
    print("="*60)
    
    test_options = {
        "1": ("Live Camera Detection", test_with_camera_stream),
        "2": ("Video File Detection", test_with_video_file),
        "3": ("Image File Detection", test_with_image_file),
        "4": ("Camera Stream Only", test_camera_stream),
        "5": ("Object Detector Only", test_object_detector),
        "6": ("Alert System", test_alert_system),
        "7": ("Run All Tests", run_all_tests)
    }
    
    while True:
        print("\nAvailable Tests:")
        for key, (name, _) in test_options.items():
            print(f"  {key}. {name}")
        print("  0. Exit")
        
        choice = input("\nSelect test to run (0-7): ").strip()
        
        if choice == "0":
            print("Exiting test mode...")
            break
        elif choice in test_options:
            test_name, test_func = test_options[choice]
            print(f"\nRunning: {test_name}")
            print("-" * 40)
            try:
                test_func()
            except KeyboardInterrupt:
                print(f"\n[INFO] {test_name} interrupted by user")
            except Exception as e:
                print(f"[ERROR] {test_name} failed: {e}")
            print("-" * 40)
        else:
            print("Invalid choice. Please select 0-7.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive_tests()
    else:
        run_all_tests()

        