import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def count_fingers(hand_landmarks):
    # Get finger tip and pip landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Get corresponding pip landmarks
    thumb_pip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    
    # Check if each finger is extended
    thumb_extended = thumb_tip.x > thumb_pip.x
    index_extended = index_tip.y < index_pip.y
    middle_extended = middle_tip.y < middle_pip.y
    ring_extended = ring_tip.y < ring_pip.y
    pinky_extended = pinky_tip.y < pinky_pip.y
    
    # Count extended fingers
    extended_fingers = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
    return sum(extended_fingers)

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize MediaPipe Pose and Hands
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose, \
        mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to read from webcam")
                continue
            
            # Flip the image horizontally for a later selfie-view display
            image = cv2.flip(image, 1)
            
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect pose and hands
            pose_results = pose.process(image_rgb)
            hands_results = hands.process(image_rgb)
            
            # Draw pose landmarks
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # Get specific landmarks
                landmarks = pose_results.pose_landmarks.landmark
                
                # Get shoulder positions
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                
                # Get hip positions
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                
                # Calculate body position
                shoulder_angle = np.arctan2(
                    right_shoulder.y - left_shoulder.y,
                    right_shoulder.x - left_shoulder.x
                ) * 180 / np.pi
                
                hip_angle = np.arctan2(
                    right_hip.y - left_hip.y,
                    right_hip.x - left_hip.x
                ) * 180 / np.pi
                
                # Determine body position
                if abs(shoulder_angle) < 10 and abs(hip_angle) < 10:
                    position = "Standing Straight"
                elif shoulder_angle > 10:
                    position = "Leaning Left"
                elif shoulder_angle < -10:
                    position = "Leaning Right"
                else:
                    position = "Unknown"
                
                # Display position on screen
                cv2.putText(image, f"Position: {position}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display angles for debugging
                cv2.putText(image, f"Shoulder Angle: {shoulder_angle:.1f}°", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"Hip Angle: {hip_angle:.1f}°", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Process and draw hand landmarks
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Count extended fingers
                    finger_count = count_fingers(hand_landmarks)
                    
                    # Get hand position
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    hand_x, hand_y = int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0])
                    
                    # Display finger count near the hand
                    cv2.putText(image, f"Fingers: {finger_count}", (hand_x, hand_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the image
            cv2.imshow('Body Pose and Hand Detection', image)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 