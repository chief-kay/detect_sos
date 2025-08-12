"""
Alert System Module
Purpose: Handles SMS alerts, message dispatch, and emergency communication via Arkesel API
"""

import requests
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
import time
from threading import Thread, Lock
import os

class AlertSystem:
    def __init__(self, api_key: str|None = "RlBsaWRPbmh1V0FFR1FMS29UQ2U", sender_id: str = "UAVDRONE"):
        """
        Initialize Alert System with Arkesel SMS API
        
        Args:
            api_key: Arkesel API key (can also be set via environment variable)
            sender_id: SMS sender ID for messages
        """
        # API Configuration
        self.api_key = api_key or os.getenv('ARKESEL_API_KEY')
        self.sender_id = sender_id
        self.base_url = "https://sms.arkesel.com/api/v2/sms/send"
        
        # Emergency contacts database
        self.emergency_contacts = {
            'police': ['233265967885', '233202184084'],
            'fire_service': ['233240000002', '233596358707'],
            'ambulance': ['233240000003', '233540000003'],
            'traffic_police': ['23327455942', '233596358707'],
            'control_center': ['233592876626', '233546327783'],
            'supervisor': ['233240000006']  # Add actual supervisor number
        }
        
        # Alert history and rate limiting
        self.alert_history = []
        self.rate_limit_lock = Lock()
        self.last_alert_time = {}
        self.min_alert_interval = 300  # 5 minutes between similar alerts
        
        # Priority mapping
        self.priority_contacts = {
            'critical': ['police', 'fire_service', 'ambulance', 'control_center'],
            'high': ['police', 'traffic_police', 'control_center'],
            'medium': ['traffic_police', 'control_center'],
            'low': ['control_center']
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Validate API configuration
        self._validate_config()

    def _validate_config(self):
        """Validate API configuration"""
        if not self.api_key:
            self.logger.error("Arkesel API key not provided. Set ARKESEL_API_KEY environment variable.")
            raise ValueError("Missing Arkesel API key")
        
        if not self.sender_id:
            self.logger.warning("No sender ID provided, using default 'UAVDRONE'")
            self.sender_id = "UAVDRONE"

    def send_sms(self, phone_numbers: List[str], message: str, priority: str = 'medium') -> Dict:
        """
        Send SMS via Arkesel API
        
        Args:
            phone_numbers: List of recipient phone numbers
            message: SMS message content
            priority: Message priority level
            
        Returns:
            API response dictionary
        """
        if not phone_numbers or not message:
            return {'success': False, 'error': 'Missing phone numbers or message'}
        
        # Prepare API payload
        payload = {
            # 'api_key': self.api_key,
            'sender': self.sender_id,
            'message': message,
            'recipients': phone_numbers
        }
        
        try:
            # Send request to Arkesel API
            response = requests.post(
                self.base_url,
                json=payload,
                headers={'Content-Type': 'application/json',
                         'api-key': self.api_key},
                timeout=30
            )
            
            response_data = response.json()
            print(f"Response from Arkesel API: {response_data}")
            
            if response_data['status'] == 'success':
                self.logger.info(f"SMS sent successfully to {len(phone_numbers)} recipients")
                return {
                    'success': True,
                    # 'message_id': response_data.get('data', {}).get('message_id'),
                    'recipients_count': len(phone_numbers),
                    'response': response_data
                }
            else:
                error_msg = response_data.get('message', 'Unknown error')
                self.logger.error(f"SMS sending failed: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'status_code': response.status_code,
                    'response': response_data
                }
                
        except requests.exceptions.Timeout:
            self.logger.error("SMS API request timed out")
            return {'success': False, 'error': 'Request timeout'}
        except requests.exceptions.RequestException as e:
            self.logger.error(f"SMS API request failed: {e}")
            return {'success': False, 'error': str(e)}
        except json.JSONDecodeError:
            self.logger.error("Invalid JSON response from SMS API")
            return {'success': False, 'error': 'Invalid API response'}

    def format_alert_message(self, event: str, location: str = "Ayeduase Road, Kumasi", 
                           drone_id: str = "4A", coordinates: str|None = None) -> str:
        """
        Creates a structured alert message for SMS
        
        Args:
            event: Description of the detected event
            location: Location where event occurred
            drone_id: Drone unit identifier
            coordinates: GPS coordinates (optional)
            
        Returns:
            Formatted alert message
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = (
            f"🚨 DRONE ALERT 🚨\n"
            f"Unit: {drone_id}\n"
            f"Time: {timestamp}\n"
            f"Event: {event}\n"
            f"Location: {location}"
        )
        
        if coordinates:
            message += f"\nGPS: {coordinates}"
            
        message += f"\nImmediate response required."
        
        return message

    def get_contacts_for_event(self, event_type: str, priority: str = 'medium') -> List[str]:
        """
        Get appropriate contacts based on event type and priority
        
        Args:
            event_type: Type of detected event
            priority: Priority level of the event
            
        Returns:
            List of phone numbers to contact
        """
        contacts = []
        
        # Get contacts based on priority
        contact_groups = self.priority_contacts.get(priority, ['control_center'])
        
        # Add specific contacts for event types
        if 'fire' in event_type.lower():
            if 'fire_service' not in contact_groups:
                contact_groups.append('fire_service')
        elif 'accident' in event_type.lower() or 'injured' in event_type.lower():
            if 'ambulance' not in contact_groups:
                contact_groups.append('ambulance')
        elif 'sos' in event_type.lower() or 'emergency' in event_type.lower():
            contact_groups = ['police', 'ambulance', 'control_center']
            
        # Collect phone numbers
        for group in contact_groups:
            if group in self.emergency_contacts:
                contacts.extend(self.emergency_contacts[group])
                
        return list(set(contacts))  # Remove duplicates

    def _should_send_alert(self, event_type: str, priority: str) -> bool:
        """
        Check if alert should be sent based on rate limiting
        
        Args:
            event_type: Type of event
            priority: Priority level
            
        Returns:
            True if alert should be sent
        """
        with self.rate_limit_lock:
            current_time = time.time()
            alert_key = f"{event_type}_{priority}"
            
            # Critical alerts always go through
            if priority == 'critical':
                return True
                
            # Check rate limiting for other priorities
            if alert_key in self.last_alert_time:
                time_since_last = current_time - self.last_alert_time[alert_key]
                if time_since_last < self.min_alert_interval:
                    self.logger.info(f"Rate limiting: {alert_key} alert suppressed")
                    return False
                    
            self.last_alert_time[alert_key] = current_time
            return True

    def trigger_alert(self, event: str, priority: str = 'medium', location: str|None = None, 
                     coordinates: str|None = None, drone_id: str = "4A") -> Dict:
        """
        High-level function to trigger emergency alert via SMS
        
        Args:
            event: Event description
            priority: Priority level ('critical', 'high', 'medium', 'low')
            location: Event location
            coordinates: GPS coordinates
            drone_id: Drone identifier
            
        Returns:
            Alert result dictionary
        """
        try:
            # Check rate limiting
            if not self._should_send_alert(event, priority):
                return {
                    'success': False,
                    'error': 'Rate limited',
                    'message': 'Alert suppressed due to rate limiting'
                }
            
            # Format alert message
            alert_message = self.format_alert_message(
                event, 
                location or "Ayeduase Road, Kumasi",
                drone_id,
                coordinates
            )
            
            # Get appropriate contacts
            contacts = self.get_contacts_for_event(event, priority)
            
            if not contacts:
                self.logger.error(f"No contacts found for event: {event}, priority: {priority}")
                return {
                    'success': False,
                    'error': 'No contacts available',
                    'message': alert_message
                }
            self.logger.info(f"Triggering {priority} alert for event: {event} to {len(contacts)} contacts with api key={self.api_key}")
            # Send SMS
            result = self.send_sms(contacts, alert_message, priority)
            
            # Log alert in history
            alert_record = {
                'timestamp': datetime.now().isoformat(),
                'event': event,
                'priority': priority,
                'location': location,
                'coordinates': coordinates,
                'drone_id': drone_id,
                'contacts_count': len(contacts),
                'success': result.get('success', False),
                'message_id': result.get('message_id')
            }
            
            self.alert_history.append(alert_record)
            
            # Console output for monitoring
            status = "✅ SENT" if result['success'] else "❌ FAILED"
            print(f"[ALERT {status}] {priority.upper()} | {event} | {len(contacts)} contacts")
            
            return {
                'success': result['success'],
                'message': alert_message,
                'contacts_notified': len(contacts),
                'sms_result': result,
                'alert_id': len(self.alert_history)
            }
            
        except Exception as e:
            self.logger.error(f"Alert trigger failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to send alert for: {event}"
            }

    def send_test_alert(self, test_numbers: List[str]|None = None) -> Dict:
        """Send a test alert to verify system functionality"""
        test_contacts = test_numbers or ['233592876626']  # Add your test number
        test_message = self.format_alert_message(
            "SYSTEM TEST - Please ignore this message",
            "Test Location",
            "TEST"
        )
        
        return self.send_sms(test_contacts, test_message, 'low')

    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """Get recent alert history"""
        return self.alert_history[-limit:] if self.alert_history else []

    def add_emergency_contact(self, group: str, phone_number: str):
        """Add new emergency contact"""
        if group not in self.emergency_contacts:
            self.emergency_contacts[group] = []
        if phone_number not in self.emergency_contacts[group]:
            self.emergency_contacts[group].append(phone_number)
            self.logger.info(f"Added contact {phone_number} to group {group}")

    def remove_emergency_contact(self, group: str, phone_number: str):
        """Remove emergency contact"""
        if group in self.emergency_contacts and phone_number in self.emergency_contacts[group]:
            self.emergency_contacts[group].remove(phone_number)
            self.logger.info(f"Removed contact {phone_number} from group {group}")

    def get_stats(self) -> Dict:
        """Get alert system statistics"""
        successful_alerts = len([a for a in self.alert_history if a['success']])
        
        return {
            'total_alerts': len(self.alert_history),
            'successful_alerts': successful_alerts,
            'failed_alerts': len(self.alert_history) - successful_alerts,
            'success_rate': (successful_alerts / len(self.alert_history) * 100) if self.alert_history else 0,
            'emergency_groups': len(self.emergency_contacts),
            'total_contacts': sum(len(contacts) for contacts in self.emergency_contacts.values())
        }


# Test and demonstration
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize alert system (you need to set your API key)
    # export ARKESEL_API_KEY="your_api_key_here"
    try:
        system = AlertSystem(api_key="RlBsaWRPbmh1V0FFR1FMS29UQ2U")
        
        # Test different priority alerts
        test_scenarios = [
            ("road accident with injuries", "low", "Ayeduase-Bomso Junction"),
            # ("vehicle overload detected", "high", "Ayeduase Road"),
            # ("illegal parking obstruction", "medium", "KNUST Main Gate"),
            # ("SOS gesture detected", "critical", "KNUST Campus")
        ]
        
        print("Testing Alert System...")
        for event, priority, location in test_scenarios:
            print(f"\nTesting {priority} alert: {event}")
            result = system.trigger_alert(event, priority, location)
            print(f"Result: {'Success' if result['success'] else 'Failed'}")
            time.sleep(2)  # Brief delay between tests
            
        # Display statistics
        stats = system.get_stats()
        print(f"\nAlert System Stats: {stats}")
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set your Arkesel API key as environment variable:")
        print("export ARKESEL_API_KEY='your_api_key_here'")