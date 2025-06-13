import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
from collections import deque

class HandGestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Control states
        self.is_controlling = False
        self.last_gesture = None
        self.gesture_cooldown = 0
        
        # Cursor smoothing
        self.cursor_history = deque(maxlen=5)  # Store last 5 positions
        self.smoothing_factor = 0.5  # Adjust this value to change smoothing (0-1)
        
        # Calibration
        self.calibration_points = []
        self.is_calibrating = False
        self.calibration_matrix = None
        
    def calibrate_cursor(self):
        """Calibrate cursor position using screen corners"""
        print("Calibration started. Please point to the four corners of your screen.")
        self.is_calibrating = True
        self.calibration_points = []
        
        # Define screen corners (in normalized coordinates)
        screen_corners = [
            (0, 0),  # Top-left
            (1, 0),  # Top-right
            (0, 1),  # Bottom-left
            (1, 1)   # Bottom-right
        ]
        
        for corner in screen_corners:
            print(f"Point to the {corner} corner of your screen and press 'c'")
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        cv2.putText(frame, f"Point to corner {corner}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Calibration', frame)
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    if results.multi_hand_landmarks:
                        index_tip = results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        self.calibration_points.append((index_tip.x, index_tip.y))
                        break
        
        self.is_calibrating = False
        cv2.destroyWindow('Calibration')
        
        # Calculate calibration matrix
        if len(self.calibration_points) == 4:
            src_points = np.array(self.calibration_points, dtype=np.float32)
            dst_points = np.array(screen_corners, dtype=np.float32)
            self.calibration_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            print("Calibration completed!")
        else:
            print("Calibration failed!")
            self.calibration_matrix = None
    
    def get_smoothed_cursor_position(self, x, y):
        """Apply smoothing to cursor movement"""
        self.cursor_history.append((x, y))
        if len(self.cursor_history) < 2:
            return x, y
            
        # Calculate weighted average of recent positions
        weights = np.linspace(0.1, 1.0, len(self.cursor_history))
        weights = weights / np.sum(weights)
        
        x_smooth = sum(x * w for (x, _), w in zip(self.cursor_history, weights))
        y_smooth = sum(y * w for (_, y), w in zip(self.cursor_history, weights))
        
        return x_smooth, y_smooth
    
    def map_to_screen(self, x, y):
        """Map camera coordinates to screen coordinates"""
        if self.calibration_matrix is not None:
            # Apply calibration transformation
            point = np.array([[[x, y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, self.calibration_matrix)
            x, y = transformed[0][0]
        
        # Map to screen coordinates
        screen_x = int(x * self.screen_width)
        screen_y = int(y * self.screen_height)
        
        # Apply smoothing
        screen_x, screen_y = self.get_smoothed_cursor_position(screen_x, screen_y)
        
        return screen_x, screen_y
    
    def is_finger_up(self, finger_landmarks, hand_landmarks):
        """Check if a finger is up based on its landmarks"""
        finger_tip = finger_landmarks[3]
        finger_pip = finger_landmarks[2]
        finger_mcp = finger_landmarks[1]
        
        # Get wrist position for reference
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        # Check if finger is extended by comparing y coordinates
        return finger_tip.y < finger_pip.y < finger_mcp.y
    
    def detect_gesture(self, hand_landmarks):
        # Get finger landmarks
        thumb = [hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC],
                hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP],
                hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP],
                hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]]
        
        index = [hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP],
                hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP],
                hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP],
                hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]]
        
        middle = [hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
                 hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                 hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP],
                 hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]]
        
        ring = [hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP],
               hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP],
               hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_DIP],
               hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]]
        
        pinky = [hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP],
                hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP],
                hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP],
                hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]]
        
        # Check which fingers are up
        thumb_up = self.is_finger_up(thumb, hand_landmarks)
        index_up = self.is_finger_up(index, hand_landmarks)
        middle_up = self.is_finger_up(middle, hand_landmarks)
        ring_up = self.is_finger_up(ring, hand_landmarks)
        pinky_up = self.is_finger_up(pinky, hand_landmarks)
        
        # Define gestures based on finger positions
        if index_up and not (middle_up or ring_up or pinky_up):
            return "point"  # Pointing with index finger
        elif index_up and middle_up and not (ring_up or pinky_up):
            return "scroll"  # Index and middle up for scrolling
        elif index_up and middle_up and ring_up and not pinky_up:
            return "right_click"  # Three fingers up for right click
        elif index_up and middle_up and ring_up and pinky_up:
            return "left_click"  # All fingers up except thumb for left click
        else:
            return None
    
    def execute_control(self, gesture, hand_landmarks):
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            return
        
        # Get index finger position and map to screen coordinates
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x, y = self.map_to_screen(index_tip.x, index_tip.y)
        
        # Move cursor
        pyautogui.moveTo(x, y)
        
        # Execute gesture-specific actions
        if gesture == "left_click":
            # Left click
            pyautogui.click()
            self.gesture_cooldown = 10
            
        elif gesture == "right_click":
            # Right click
            pyautogui.rightClick()
            self.gesture_cooldown = 10
            
        elif gesture == "scroll":
            # Scroll using middle finger position relative to index
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            scroll_amount = int((middle_tip.y - index_tip.y) * 100)
            pyautogui.scroll(scroll_amount)
    
    def run(self):
        print("Finger Gesture Control System Started!")
        print("Gestures:")
        print("- All fingers up (except thumb): Left click")
        print("- Three fingers up (index, middle, ring): Right click")
        print("- Two fingers up (index, middle): Scroll")
        print("- One finger up (index): Point mode")
        print("Cursor will always follow your index finger")
        print("Press 'c' to calibrate")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect hands
            results = self.hands.process(frame_rgb)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Detect and execute gesture
                    gesture = self.detect_gesture(hand_landmarks)
                    if gesture:
                        self.execute_control(gesture, hand_landmarks)
                        # Display current gesture
                        cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Finger Gesture Control', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.calibrate_cursor()
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandGestureController()
    controller.run() 