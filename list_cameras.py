import cv2

def list_available_cameras():
    # Try to open cameras from index 0 to 9
    available_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i} is available")
                # Get camera properties
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"Resolution: {width}x{height}, FPS: {fps}")
                available_cameras.append(i)
            cap.release()
    
    if not available_cameras:
        print("No cameras found!")
    else:
        print(f"\nTotal available cameras: {len(available_cameras)}")
        print("Available camera indices:", available_cameras)

if __name__ == "__main__":
    list_available_cameras() 