import cv2
import mediapipe as mp
import numpy as np

def test_face_detection():
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
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
    
    # Initialize face detection
    with mp_face_detection.FaceDetection(
        model_selection=1,  # Use full range model
        min_detection_confidence=0.5) as face_detection:
        
        print("Face detection started!")
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
            
            # Process both frames
            results0 = face_detection.process(frame0_rgb)
            results1 = face_detection.process(frame1_rgb)
            
            # Draw face detections
            if results0.detections:
                for detection in results0.detections:
                    mp_drawing.draw_detection(frame0, detection)
                    # Add confidence score
                    conf = detection.score[0]
                    cv2.putText(frame0, f"Conf: {conf:.2f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 255, 0), 2)
            
            if results1.detections:
                for detection in results1.detections:
                    mp_drawing.draw_detection(frame1, detection)
                    # Add confidence score
                    conf = detection.score[0]
                    cv2.putText(frame1, f"Conf: {conf:.2f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 255, 0), 2)
            
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
            cv2.imshow('Face Detection Test - Press q to quit', combined)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release the cameras
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_face_detection() 