import cv2
import mediapipe as mp
import numpy as np
from enum import Enum

class ExerciseType(Enum):
    SQUAT = "Squat"
    LUNGE = "Lunge"
    PUSHUP = "Push-up"
    JUMPING_JACK = "Jumping Jack"
    NONE = "None"

class ExerciseTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Exercise state variables
        self.current_exercise = ExerciseType.NONE
        self.exercise_count = 0
        self.exercise_stage = None
        self.angle_threshold = 90
        
        # Squat detection parameters
        self.squat_min_angle = 70  # Minimum angle for squat
        self.squat_max_angle = 160  # Maximum angle for squat
        self.squat_hold_time = 0  # Time spent in squat position
        self.squat_hold_threshold = 10  # Frames to hold for valid squat
        self.last_hip_y = None  # Track vertical movement
        self.vertical_movement_threshold = 0.1  # Minimum vertical movement required
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
        return angle
    
    def detect_squat(self, landmarks):
        """Detect and count squats without hold time"""
        # Get hip, knee, and ankle positions
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        # Calculate knee angle
        angle = self.calculate_angle(hip, knee, ankle)
        
        # Track vertical movement
        current_hip_y = hip[1]
        vertical_movement = 0
        if self.last_hip_y is not None:
            vertical_movement = abs(current_hip_y - self.last_hip_y)
        self.last_hip_y = current_hip_y
        
        # Check if the person is in a squat position
        is_squat_position = self.squat_min_angle <= angle <= self.squat_max_angle
        
        # Count squat only if:
        # 1. Person is in squat position (down)
        # 2. Has sufficient vertical movement
        if is_squat_position:
            if self.exercise_stage == 'up' and vertical_movement >= self.vertical_movement_threshold:
                self.exercise_stage = 'down'
        else:
            if self.exercise_stage == 'down':
                self.exercise_stage = 'up'
                self.exercise_count += 1
        
        return angle
    
    def detect_pushup(self, landmarks):
        """Detect and count push-ups"""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        angle = self.calculate_angle(shoulder, elbow, wrist)
        
        if angle > 160 and self.exercise_stage == 'down':
            self.exercise_stage = 'up'
        elif angle < 90 and self.exercise_stage == 'up':
            self.exercise_stage = 'down'
            self.exercise_count += 1
            
        return angle
    
    def process_frame(self, frame):
        """Process a single frame and detect exercises"""
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = self.pose.process(image)
        
        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Check if any pose landmarks were detected
        if not results.pose_landmarks:
            # Display message when no pose is detected
            cv2.putText(image, 'No pose detected',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return image
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Initialize exercise stage if not set
            if self.exercise_stage is None:
                self.exercise_stage = 'up'
            
            # Detect exercise based on current exercise type
            if self.current_exercise == ExerciseType.SQUAT:
                angle = self.detect_squat(landmarks)
            elif self.current_exercise == ExerciseType.PUSHUP:
                angle = self.detect_pushup(landmarks)
            else:
                angle = 0
            
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Display exercise information
            cv2.putText(image, f'Exercise: {self.current_exercise.value}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'Count: {self.exercise_count}',
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'Angle: {angle:.2f}',
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display additional squat information
            if self.current_exercise == ExerciseType.SQUAT:
                cv2.putText(image, f'Hold Time: {self.squat_hold_time}',
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Display error message on frame
            cv2.putText(image, 'Error processing pose',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return image

def main():
    cap = cv2.VideoCapture(0)
    tracker = ExerciseTracker()
    
    # Set initial exercise type
    tracker.current_exercise = ExerciseType.SQUAT
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame = tracker.process_frame(frame)
        
        # Display the frame
        cv2.imshow('Exercise Tracker', processed_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 