"""
Object Detector Module for UAV Traffic Surveillance Drone
Purpose: Detects vehicles, people, and violations using custom trained YOLOv5 model
         Determines emergency severity and logs/report actions.
"""

import torch
import cv2
import numpy as np
from datetime import datetime
import logging
import os
from typing import List, Dict, Optional, Tuple
import time
from ultralytics import YOLO

# Import the AlertSystem
from .alert_system import AlertSystem

class ObjectDetector:
    def __init__(self, model_path='resources/best.pt', conf_thresh=0.4, device='auto', 
                 enable_alerts=True, api_key=None):
        """
        Initialize object detector with custom trained YOLOv5 model.
        
        Args:
            model_path: Path to custom YOLOv5 weights file
            conf_thresh: Confidence threshold for detections
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            enable_alerts: Enable SMS alert system
            api_key: Arkesel API key for SMS alerts
        """
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.device = device
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize Alert System
        self.enable_alerts = enable_alerts
        self.alert_system = None
        if enable_alerts:
            try:
                self.alert_system = AlertSystem(api_key=api_key)
                self.logger.info("Alert system initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize alert system: {e}")
                self.enable_alerts = False

        # Custom class names from training YAML
        self.class_names = {
            # Cars & Trucks
            0: 'truck_xl', 1: 'truck_s', 2: 'truck_m', 3: 'truck_l',
            4: 'small_truck', 5: 'small_bus', 6: 'mid_truck', 7: 'car',
            8: 'bus_s', 9: 'bus_l', 10: 'big_truck', 11: 'big_bus',
            
            # People
            12: 'person',
            
            # Motorbikes
            13: 'motorbike',
            
            # Gestures
            14: 'sos_gesture', 15: 'not_sos_gesture', 16: 'emergency_call_for_help',
            
            # Fire types
            17: 'fire_solid', 18: 'fire_metal', 19: 'fire_liquid',
            
            # Tricycle
            20: 'tricycle',
            
            # Minibus types
            21: 'minibus_overload', 22: 'minibus_normal', 23: 'minibus',
            
            # Helmet
            24: 'without_helmet', 25: 'helmet_2', 26: 'helmet',
            
            # License plate
            27: 'license_plate',
            
            # Injured people
            28: 'not_injured', 29: 'injured',
            
            # Road accident
            30: 'accident'
        }
        
        # Initialize model
        self.model = None
        self._load_model()
        
        # Detection tracking
        self.violation_log = []
        self.detection_count = 0
        self.last_detection_time = None
        
        # Alert rate limiting (additional layer beyond AlertSystem)
        self.last_alert_sent = {}
        self.alert_cooldown = {
            'critical': 60,    # 1 minute cooldown for critical
            'high': 180,       # 3 minutes for high priority
            'medium': 300,     # 5 minutes for medium priority
            'low': 600         # 10 minutes for low priority
        }
        
        # Configuration
        self.road_speed_limit = 50  # km/h default for Ghanaian urban roads
        self.last_congestion_time = None
        self.traffic_congestion_timer = 180  # seconds
        
        # Event priorities based on actual classes
        self.critical_events = [
            'accident', 'fire_solid', 'fire_metal', 'fire_liquid', 
            'sos_gesture', 'emergency_call_for_help', 'injured'
        ]
        
        self.high_priority_events = [
            'minibus_overload', 'without_helmet', 'person'
        ]
        
        self.medium_priority_events = [
            'not_sos_gesture', 'not_injured'
        ]
        
        # Vehicle categories for logic processing
        self.vehicles = [
            'truck_xl', 'truck_s', 'truck_m', 'truck_l', 'small_truck',
            'small_bus', 'mid_truck', 'car', 'bus_s', 'bus_l', 
            'big_truck', 'big_bus', 'tricycle', 'minibus_overload',
            'minibus_normal', 'minibus', 'motorbike'
        ]
        
        self.fire_types = ['fire_solid', 'fire_metal', 'fire_liquid']
        self.helmet_types = ['helmet', 'helmet_2', 'without_helmet']
        self.gesture_types = ['sos_gesture', 'not_sos_gesture', 'emergency_call_for_help']
        
        # Colors for visualization (BGR format) - Tableau 10 color scheme with custom mapping
        self.bbox_colors = [
            (164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
            (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184),
            (255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255),
            (0,255,255), (128,0,0), (0,128,0), (0,0,128), (128,128,0),
            (128,0,128), (0,128,128), (255,128,0), (255,0,128), (128,255,0),
            (0,255,128), (128,0,255), (0,128,255), (255,255,128), (255,128,255),
            (128,255,255)
        ]
        
        
    def _load_model(self):
        """Load custom trained YOLO model with error handling."""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                self.logger.error(f"Custom model file not found at {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load custom model using ultralytics YOLO
            self.model = YOLO(self.model_path, task='detect')
            
            # Get model labels
            self.labels = self.model.names
            
            self.logger.info(f"Custom model loaded successfully")
            self.logger.info(f"Model classes: {len(self.labels)}")
            self.logger.info(f"Available classes: {list(self.labels.values())}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _should_send_alert(self, event_type: str, priority: str) -> bool:
        """
        Additional rate limiting check for alerts (detector level)
        
        Args:
            event_type: Type of event detected
            priority: Priority level of the event
            
        Returns:
            True if alert should be sent
        """
        current_time = time.time()
        alert_key = f"{event_type}_{priority}"
        
        # Critical alerts always go through (minimal cooldown)
        if priority == 'critical':
            if alert_key in self.last_alert_sent:
                time_since_last = current_time - self.last_alert_sent[alert_key]
                if time_since_last < self.alert_cooldown['critical']:
                    return False
        else:
            # Check cooldown for other priorities
            if alert_key in self.last_alert_sent:
                time_since_last = current_time - self.last_alert_sent[alert_key]
                if time_since_last < self.alert_cooldown.get(priority, 300):
                    return False
        
        self.last_alert_sent[alert_key] = current_time
        return True

    def _send_alert(self, detection: Dict, location: str|None = None, coordinates: str|None = None):
        """
        Send alert for detected violation
        
        Args:
            detection: Detection dictionary with event details
            location: Location information
            coordinates: GPS coordinates
        """
        if not self.enable_alerts or not self.alert_system:
            return
            
        event_type = detection.get('event', 'unknown')
        priority = detection.get('priority', 'low')
        
        # Check if we should send this alert (rate limiting)
        if not self._should_send_alert(event_type, priority):
            self.logger.debug(f"Alert rate limited: {event_type} ({priority})")
            return
        
        # Format event description
        event_description = self._format_event_description(detection)
        
        try:
            # Send alert via AlertSystem
            result = self.alert_system.trigger_alert(
                event=event_description,
                priority=priority,
                location=location or "Ayeduase Road, Kumasi",
                coordinates=coordinates,
                drone_id="UAV-4A"
            )
            
            if result.get('success'):
                self.logger.info(f"Alert sent successfully for {event_type}")
                detection['alert_sent'] = True
                detection['alert_id'] = result.get('alert_id')
            else:
                self.logger.error(f"Alert failed for {event_type}: {result.get('error')}")
                detection['alert_sent'] = False
                detection['alert_error'] = result.get('error')
                
        except Exception as e:
            self.logger.error(f"Exception sending alert for {event_type}: {e}")
            detection['alert_sent'] = False
            detection['alert_error'] = str(e)

    def _format_event_description(self, detection: Dict) -> str:
        """
        Format event description for alerts
        
        Args:
            detection: Detection dictionary
            
        Returns:
            Formatted event description
        """
        label = detection['label']
        confidence = detection['confidence']
        event = detection.get('event', 'unknown')
        
        # Create descriptive message based on event type
        descriptions = {
            'road_accident': f"Road accident detected with {confidence:.0%} confidence",
            'fire_detected': f"Fire hazard detected ({label}) with {confidence:.0%} confidence",
            'sos_signal': f"SOS gesture detected - person requesting emergency help with {confidence:.0%} confidence",
            'help_requested': f"Emergency call for help detected with {confidence:.0%} confidence",
            'injured_person': f"Injured person detected requiring immediate medical attention with {confidence:.0%} confidence",
            'vehicle_overload': f"Overloaded minibus detected violating capacity regulations with {confidence:.0%} confidence",
            'helmet_violation': f"Motorcycle rider without helmet detected with {confidence:.0%} confidence",
            'pedestrian_in_road': f"Pedestrian detected in roadway creating traffic hazard with {confidence:.0%} confidence",
            'illegal_parking': f"Illegally parked {label} causing traffic obstruction with {confidence:.0%} confidence",
            'license_detected': f"License plate detected for automated processing with {confidence:.0%} confidence",
            'normal_gesture': f"Non-emergency gesture detected with {confidence:.0%} confidence"
        }
        
        return descriptions.get(event, f"{event} detected: {label} with {confidence:.0%} confidence")

    def detect(self, frame: np.ndarray, location: str | None = None, coordinates: str | None = None) -> List[Dict]:
        """
        Detect objects in frame and return structured results.
        
        Args:
            frame: Input image frame
            location: Current location for alerts
            coordinates: GPS coordinates for alerts
            
        Returns:
            List of detection dictionaries
        """
        if frame is None or self.model is None:
            return []
            
        try:
            # Run inference with tracking enabled for consistency
            results = self.model.track(frame, verbose=False, conf=self.conf_thresh)
            
            # Extract detections
            detections = results[0].boxes
            output = []
            
            if detections is None or len(detections) == 0:
                return output
            
            # First pass: create all detection objects
            for i in range(len(detections)):
                try:
                    # Get bounding box coordinates
                    xyxy_tensor = detections[i].xyxy.cpu() # type: ignore
                    xyxy = xyxy_tensor.numpy().squeeze()
                    x1, y1, x2, y2 = xyxy.astype(int)
                    
                    # Get class information
                    class_id = int(detections[i].cls.item())
                    label = self.labels.get(class_id, f'unknown_{class_id}')
                    
                    # Get confidence
                    confidence = float(detections[i].conf.item())
                    
                    # Skip if confidence too low
                    if confidence < self.conf_thresh:
                        continue
                    
                    # Create detection dictionary
                    detection = {
                        'label': label,
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                        'area': (x2 - x1) * (y2 - y1),
                        'requires_action': False,
                        'priority': 'low',
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'alert_sent': False
                    }
                    
                    output.append(detection)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing detection {i}: {e}")
                    continue
            
            # Second pass: apply detection logic with full context
            for detection in output:
                try:
                    self._apply_detection_logic(frame, detection, output)
                    
                    # Log violations and send alerts if any
                    if detection.get('event') and detection['event'] != 'road_surface':
                        self.log_violation(detection)
                        
                        # Send alert if required and enabled
                        if detection.get('requires_action') and self.enable_alerts:
                            self._send_alert(detection, location, coordinates)
                            
                except Exception as e:
                    self.logger.warning(f"Error applying logic to detection: {e}")
                    continue
            
            # Analyze road context if road is detected
            if self.check_road_in_detections(output):
                road_analysis = self.analyze_objects_on_road(output)
                self.logger.info(f"Road analysis: {len(road_analysis['objects_on_road'])} objects on road, "
                               f"{len(road_analysis['persons_on_road'])} persons, "
                               f"{len(road_analysis['violations_on_road'])} violations")
            
            # Update statistics
            self.detection_count += len(output)
            self.last_detection_time = time.time()
            
            return output
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []
    
    def analyze_objects_on_road(self, detections: List[Dict]) -> Dict:
        """
        Analyze which objects are positioned on detected road areas.
        
        Args:
            detections: List of all detections including road
            
        Returns:
            Analysis results dictionary
        """
        road_detections = self.get_road_detections(detections)
        
        if not road_detections:
            return {
                'has_road': False,
                'objects_on_road': [],
                'persons_on_road': [],
                'vehicles_on_road': [],
                'violations_on_road': []
            }
        
        objects_on_road = []
        persons_on_road = []
        vehicles_on_road = []
        violations_on_road = []
        
        # Check each non-road detection against road areas
        for detection in detections:
            if detection['label'] == 'road':
                continue
                
            # Check if object overlaps with any road detection
            for road_det in road_detections:
                if self._bbox_overlap(detection['bbox'], road_det['bbox'], threshold=0.1):
                    objects_on_road.append(detection)
                    
                    # Categorize objects on road
                    if detection['label'] == 'person':
                        persons_on_road.append(detection)
                    elif detection['label'] in self.vehicles:
                        vehicles_on_road.append(detection)
                    
                    # Check for violations
                    if detection.get('requires_action') or detection.get('event'):
                        violations_on_road.append(detection)
                    
                    break  # Object found on road, no need to check other road areas
        
        return {
            'has_road': True,
            'road_count': len(road_detections),
            'objects_on_road': objects_on_road,
            'persons_on_road': persons_on_road,
            'vehicles_on_road': vehicles_on_road,
            'violations_on_road': violations_on_road,
            'road_areas': road_detections
        }

    def _bbox_overlap(self, bbox1: List[int], bbox2: List[int], threshold: float = 0.1) -> bool:
        """
        Check if two bounding boxes overlap.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            threshold: Minimum overlap ratio to consider as overlapping
            
        Returns:
            True if bounding boxes overlap above threshold
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate overlap ratio
        overlap_ratio = intersection_area / min(area1, area2)
        
        return overlap_ratio > threshold
    
    def _person_in_roadway(self, frame: np.ndarray, bbox: List[int], detections: List[Dict] | None = None) -> bool:
        """
        Enhanced logic to detect if person is in roadway using road detections.
        
        Args:
            frame: Input frame
            bbox: Person's bounding box
            detections: All current detections (to check for road)
            
        Returns:
            True if person is detected on road area
        """
        if detections:
            # Use actual road detections if available
            road_detections = self.get_road_detections(detections)
            if road_detections:
                for road_det in road_detections:
                    if self._bbox_overlap(bbox, road_det['bbox'], threshold=0.2):
                        return True
                return False
        
        # Fallback to heuristic method if no road detections
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Get person center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Heuristic: assume roadway is in the center-bottom portion of the frame
        roadway_top = height * 0.4
        roadway_bottom = height * 0.9
        roadway_left = width * 0.1
        roadway_right = width * 0.9
        
        in_roadway_area = (roadway_left < center_x < roadway_right and 
                          roadway_top < center_y < roadway_bottom)
        
        return in_roadway_area


    def detect_illegal_parking(self, frame: np.ndarray, bbox: List[int], detections: List[Dict] | None = None) -> bool:
        """
        Enhanced illegal parking detection using road detections.
        
        Args:
            frame: Input frame
            bbox: Vehicle's bounding box
            detections: All current detections (to check for road)
            
        Returns:
            True if vehicle appears to be illegally parked
        """
        if detections:
            # Use actual road detections if available
            road_detections = self.get_road_detections(detections)
            if road_detections:
                for road_det in road_detections:
                    if self._bbox_overlap(bbox, road_det['bbox'], threshold=0.3):
                        # Vehicle is on road - could be illegally parked
                        # Additional logic could check if vehicle is stationary
                        x1, y1, x2, y2 = bbox
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Large vehicle taking up road space
                        if width > 80 and height > 60:
                            return True
                return False
        
        # Fallback to original heuristic method
        return False

    def _apply_detection_logic(self, frame: np.ndarray, detection: Dict, all_detections: List[Dict] | None = None):
        """Apply custom logic based on detected object type."""
        label = detection['label']
        bbox = detection['bbox']
        
        # Critical emergency events
        if label in self.critical_events:
            detection['requires_action'] = True
            detection['priority'] = 'critical'
            
            if label == 'accident':
                detection['event'] = 'road_accident'
                detection['action'] = 'immediate_response'
            elif label in self.fire_types:
                detection['event'] = 'fire_detected'
                detection['action'] = 'fire_response'
            elif label == 'sos_gesture':
                detection['event'] = 'sos_signal'
                detection['action'] = 'emergency_response'
            elif label == 'emergency_call_for_help':
                detection['event'] = 'help_requested'
                detection['action'] = 'emergency_response'
            elif label == 'injured':
                detection['event'] = 'injured_person'
                detection['action'] = 'medical_response'
                
        # High priority events
        elif label in self.high_priority_events:
            detection['priority'] = 'high'
            
            if label == 'minibus_overload':
                detection['event'] = 'vehicle_overload'
                detection['requires_action'] = True
                detection['action'] = 'traffic_enforcement'
            elif label == 'without_helmet':
                detection['event'] = 'helmet_violation'
                detection['requires_action'] = True
                detection['action'] = 'safety_enforcement'
            elif label == 'person':
                # Check if person is in roadway using road detections
                if self._person_in_roadway(frame, bbox, all_detections):
                    detection['event'] = 'pedestrian_in_road'
                    detection['requires_action'] = True
                    detection['priority'] = 'high'
                    
        # Vehicle analysis
        elif label in self.vehicles:
            # Check for illegal parking using road detections
            if self.detect_illegal_parking(frame, bbox, all_detections):
                detection['event'] = 'illegal_parking'
                detection['requires_action'] = True
                detection['priority'] = 'medium'
                
        # Road detection
        elif label == 'road':
            detection['event'] = 'road_surface'
            detection['priority'] = 'low'
            detection['action'] = 'spatial_reference'
                
        # License plate detection
        elif label == 'license_plate':
            detection['event'] = 'license_detected'
            detection['action'] = 'ocr_processing'
            
        # Non-SOS gesture
        elif label == 'not_sos_gesture':
            detection['event'] = 'normal_gesture'
            detection['priority'] = 'low'

    def log_violation(self, detection: Dict):
        """Log violation with timestamp and details."""
        priority_emoji = {
            'critical': '🚨',
            'high': '⚠️',
            'medium': '⚡',
            'low': 'ℹ️'
        }
        
        emoji = priority_emoji.get(detection.get('priority', 'low'), 'ℹ️')
        
        log_msg = (f"{emoji} [VIOLATION {detection['timestamp']}] "
                  f"Event: {detection.get('event', 'unknown')} | "
                  f"Label: {detection['label']} | "
                  f"Confidence: {detection['confidence']:.2f} | "
                  f"Priority: {detection.get('priority', 'low').upper()}")
        
        if detection.get('action'):
            log_msg += f" | Action: {detection['action']}"
            
        if detection.get('alert_sent'):
            log_msg += " | Alert: SENT"
        elif detection.get('requires_action'):
            log_msg += " | Alert: PENDING"
        
        print(log_msg)  # Console output
        self.logger.info(log_msg)  # File logging
        self.violation_log.append(detection)

    def get_road_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Get all road detections from the detection list.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of road detections
        """
        road_detections = []
        for detection in detections:
            if detection['label'] == 'road':
                road_detections.append(detection)
        return road_detections

    def check_road_in_detections(self, detections: List[Dict]) -> bool:
        """
        Check if 'road' label exists in current detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            True if road is detected, False otherwise
        """
        for detection in detections:
            if detection['label'] == 'road':
                return True
        return False

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame with priority indicators.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            
        Returns:
            Frame with drawn detections
        """
        if not detections:
            return frame
            
        frame_copy = frame.copy()
        print("Drawing detections on frame...", detections)
        
        for detection in detections:
            label = detection['label']
            confidence = detection['confidence']
            priority = detection.get('priority', 'low')
            class_id = detection['class_id']
            x1, y1, x2, y2 = detection['bbox']
            
            # Get color for this class using color palette
            color = self.bbox_colors[class_id % len(self.bbox_colors)]
            
            # Adjust thickness based on priority
            thickness = {'critical': 4, 'high': 3, 'medium': 2, 'low': 2}[priority]
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            label_text = f"{label}: {int(confidence*100)}%"
            if detection.get('event'):
                label_text += f" [{detection['event']}]"
                
            # Calculate text size and position
            font_scale = 0.6 if priority in ['critical', 'high'] else 0.5
            labelSize, baseLine = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            
            # Make sure not to draw label too close to top of window
            label_ymin = max(y1, labelSize[1] + 10)
            
            # Draw label background
            cv2.rectangle(frame_copy, 
                         (x1, label_ymin - labelSize[1] - 10),
                         (x1 + labelSize[0], label_ymin + baseLine - 10),
                         color, cv2.FILLED)
            
            # Draw label text
            cv2.putText(frame_copy, label_text, (x1, label_ymin - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
            
            # Draw priority indicator
            if priority == 'critical':
                cv2.circle(frame_copy, (x2 - 15, y1 + 15), 10, (0, 0, 255), -1)
                cv2.putText(frame_copy, "!", (x2 - 20, y1 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            elif priority == 'high':
                cv2.circle(frame_copy, (x2 - 15, y1 + 15), 8, (0, 165, 255), -1)
            
            # Draw alert indicator
            if detection.get('alert_sent'):
                cv2.circle(frame_copy, (x1 + 15, y1 + 15), 6, (0, 255, 0), -1)  # Green circle for sent alert
            elif detection.get('requires_action'):
                cv2.circle(frame_copy, (x1 + 15, y1 + 15), 6, (0, 255, 255), -1)  # Yellow circle for pending alert
                
        return frame_copy

    def get_violation_log(self) -> List[Dict]:
        """Get copy of violation log."""
        return self.violation_log.copy()
        
    def get_stats(self) -> Dict:
        """Get detector statistics."""
        violation_counts = {}
        alert_counts = {'sent': 0, 'failed': 0, 'pending': 0}
        
        for violation in self.violation_log:
            priority = violation.get('priority', 'low')
            violation_counts[priority] = violation_counts.get(priority, 0) + 1
            
            if violation.get('alert_sent'):
                alert_counts['sent'] += 1
            elif violation.get('alert_error'):
                alert_counts['failed'] += 1
            elif violation.get('requires_action'):
                alert_counts['pending'] += 1
            
        stats = {
            'total_detections': self.detection_count,
            'total_violations': len(self.violation_log),
            'violation_by_priority': violation_counts,
            'alert_statistics': alert_counts,
            'last_detection_time': self.last_detection_time,
            'model_path': self.model_path,
            'confidence_threshold': self.conf_thresh,
            'classes_count': len(self.labels) if self.labels else 0,
            'alerts_enabled': self.enable_alerts
        }
        
        # Add alert system stats if available
        if self.alert_system:
            stats['alert_system_stats'] = self.alert_system.get_stats()
            
        return stats

    def clear_logs(self):
        """Clear violation logs."""
        self.violation_log.clear()
        self.last_alert_sent.clear()

    def toggle_alerts(self, enable: bool):
        """Enable or disable alert system"""
        self.enable_alerts = enable and self.alert_system is not None
        self.logger.info(f"Alerts {'enabled' if self.enable_alerts else 'disabled'}")

    def send_test_alert(self):
        """Send a test alert to verify system functionality"""
        if not self.enable_alerts or not self.alert_system:
            return {'success': False, 'error': 'Alert system not available'}
            
        return self.alert_system.send_test_alert()


# Video stream test integration with alerts
def test_with_camera_stream():
    """Test object detector with camera stream and alerts."""
    import sys
    import os
    
    # Add parent directory to path to import camera_stream
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    try:
        from core.camera_stream import CameraStream
    except ImportError:
        print("[ERROR] Could not import CameraStream. Make sure the file exists.")
        return
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components with alerts enabled
    try:
        # You can disable alerts by setting enable_alerts=False
        detector = ObjectDetector(enable_alerts=True, api_key="RlBsaWRPbmh1V0FFR1FMS29UQ2U")
        camera = CameraStream(width=640, height=480, fps=30)
        
        # Start camera
        camera.start()
        print("[INFO] Camera started. Controls:")
        print("  'q' - Quit")
        print("  's' - Show stats") 
        print("  'c' - Clear logs")
        print("  'p' - Save screenshot")
        print("  'a' - Toggle alerts")
        print("  't' - Send test alert")
        print("[INFO] Priority levels: 🚨 Critical | ⚠️ High | ⚡ Medium | ℹ️ Low")
        print("[INFO] Alert indicators: 🟢 Sent | 🟡 Pending | 🔴 Failed")
        
        frame_count = 0
        
        while True:
            # Read frame
            frame = camera.read()
            if frame is None:
                continue
                
            frame_count += 1
            
            # Detect objects with location info for alerts
            detections = detector.detect(
                frame, 
                location="KNUST Campus, Ayeduase Road",
                coordinates="6.6745° N, 1.5716° W"
            )
            
            # Draw detections
            annotated_frame = detector.draw_detections(frame, detections)
            
            # Add info overlay
            critical_count = len([d for d in detections if d.get('priority') == 'critical'])
            high_count = len([d for d in detections if d.get('priority') == 'high'])
            alerts_sent = len([d for d in detections if d.get('alert_sent')])
            
            # Draw info rectangle
            cv2.rectangle(annotated_frame, (10, 10), (600, 120), (50,50,50), cv2.FILLED)
            cv2.putText(annotated_frame, f'Frame: {frame_count} | Detections: {len(detections)}', 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(annotated_frame, f'Critical: {critical_count} | High: {high_count}', 
                       (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,102,51), 1)
            cv2.putText(annotated_frame, f'Total violations: {len(detector.violation_log)} | Alerts sent: {alerts_sent}', 
                       (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (51,204,51), 1)
            cv2.putText(annotated_frame, f'Alert System: {"ON" if detector.enable_alerts else "OFF"}', 
                       (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if detector.enable_alerts else (0,0,255), 1)
            
            # Show frame
            cv2.imshow("UAV Surveillance Object Detection with Alerts", annotated_frame)
            
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
                cv2.imwrite(f'detection_capture_{frame_count}.png', annotated_frame)
                print(f"[INFO] Screenshot saved as detection_capture_{frame_count}.png")
            elif key == ord('a') or key == ord('A'):
                detector.toggle_alerts(not detector.enable_alerts)
                print(f"[INFO] Alerts {'enabled' if detector.enable_alerts else 'disabled'}")
            elif key == ord('t') or key == ord('T'):
                result = detector.send_test_alert()
                print(f"[INFO] Test alert result: {result}")
                
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


if __name__ == "__main__":
    test_with_camera_stream()