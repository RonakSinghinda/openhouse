import cv2
import numpy as np

def test_both_cameras():
    # Open both cameras
    cap0 = cv2.VideoCapture(0)  # Built-in webcam
    cap1 = cv2.VideoCapture(1)  # DroidCam
    
    if not cap0.isOpened() or not cap1.isOpened():
        print("Error: Could not open one or both cameras")
        return
    
    print("Both cameras opened successfully!")
    print("Press 'q' to quit")
    
    while True:
        # Read frames from both cameras
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if not ret0 or not ret1:
            print("Error: Could not read frames")
            break
        
        # Resize frames to be the same size
        frame0 = cv2.resize(frame0, (640, 480))
        frame1 = cv2.resize(frame1, (640, 480))
        
        # Create a combined display
        combined = np.hstack((frame0, frame1))
        
        # Add labels
        cv2.putText(combined, "Camera 0 (Webcam)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Camera 1 (DroidCam)", (650, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the combined frame
        cv2.imshow('Camera Test - Press q to quit', combined)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the cameras
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_both_cameras() 