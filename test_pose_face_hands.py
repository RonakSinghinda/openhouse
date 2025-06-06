import cv2
import mediapipe as mp
import numpy as np

def test_pose_face_hands():
    # Initialize MediaPipe solutions
    mp_pose = mp.solutions.pose
    mp_face_detection = mp.solutions.face_detection
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Initialize both cameras
    cap0 = cv2.VideoCapture(0)  # Built-in webcam
    cap1 = cv2.VideoCapture(1)  # DroidCam
    
    if not cap0.isOpened() or not cap1.isOpened():
        print("Error: Could not open one or both cameras")
        return
    
    # Set camera properties for better quality
    for cap in [cap0, cap1]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize pose, face detection, and hands
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose, \
        mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5) as face_detection, \
        mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        
        print("Pose, face, and hand detection started!")
        print("Press 'q' to quit")
        
        while True:
            # Read frames from both cameras
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            
            if not ret0 or not ret1:
                print("Error: Could not read frames")
                break
            
            # Convert to RGB for MediaPipe
            frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            
            # Process both frames for pose
            results0_pose = pose.process(frame0_rgb)
            results1_pose = pose.process(frame1_rgb)
            
            # Process both frames for face
            results0_face = face_detection.process(frame0_rgb)
            results1_face = face_detection.process(frame1_rgb)
            
            # Process both frames for hands
            results0_hands = hands.process(frame0_rgb)
            results1_hands = hands.process(frame1_rgb)
            
            # Draw pose landmarks
            if results0_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame0,
                    results0_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            if results1_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame1,
                    results1_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Draw face detections
            if results0_face.detections:
                for detection in results0_face.detections:
                    mp_drawing.draw_detection(frame0, detection)
            
            if results1_face.detections:
                for detection in results1_face.detections:
                    mp_drawing.draw_detection(frame1, detection)
            
            # Draw hand landmarks
            if results0_hands.multi_hand_landmarks:
                for hand_landmarks in results0_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame0,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            
            if results1_hands.multi_hand_landmarks:
                for hand_landmarks in results1_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame1,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            
            # Add labels
            cv2.putText(frame0, "Camera 0 (Webcam)", (10, frame0.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame1, "Camera 1 (DroidCam)", (10, frame1.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Resize frames to be the same size
            frame0 = cv2.resize(frame0, (640, 480))
            frame1 = cv2.resize(frame1, (640, 480))
            
            # Create a combined display
            combined = np.hstack((frame0, frame1))
            
            # Display the combined frame
            cv2.imshow('Pose, Face, and Hand Detection - Press q to quit', combined)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release the cameras
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_pose_face_hands() 