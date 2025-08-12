"""
flight_control.py
Purpose: Defines basic flight commands for manual and autonomous modes
"""

def takeoff():
    print("[FLIGHT CONTROL] Takeoff command sent.")

def land():
    print("[FLIGHT CONTROL] Landing command sent.")

def move_forward():
    print("[FLIGHT CONTROL] Moving forward.")

def move_backward():
    print("[FLIGHT CONTROL] Moving backward.")

def move_left():
    print("[FLIGHT CONTROL] Moving left.")

def move_right():
    print("[FLIGHT CONTROL] Moving right.")

def yaw_left():
    print("[FLIGHT CONTROL] Rotating (yaw) left.")

def yaw_right():
    print("[FLIGHT CONTROL] Rotating (yaw) right.")

def hover():
    print("[FLIGHT CONTROL] Hovering in place.")

if __name__ == "__main__":
    # Test calls
    takeoff()
    move_forward()
    hover()
    land()
