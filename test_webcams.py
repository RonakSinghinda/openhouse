import cv2

def test_webcams():
    # Try to open webcams 0 and 1
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)
    
    if not cap0.isOpened():
        print("Error: Could not open webcam 0")
        return False
    
    if not cap1.isOpened():
        print("Error: Could not open webcam 1")
        return False
    
    print("Successfully opened both webcams!")
    
    # Release the cameras
    cap0.release()
    cap1.release()
    return True

if __name__ == "__main__":
    test_webcams() 