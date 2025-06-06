import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from utils import DLT, get_projection_matrix, write_keypoints_to_disk

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection

frame_shape = [720, 1280]

#add here if you need more keypoints
pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]

def run_mp(input_stream1, input_stream2, P0, P1):
    #input video stream
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    #set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    #create body keypoints detector objects
    pose0 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose1 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    #create hand detector objects
    hands0 = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    hands1 = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    #create face detector objects
    face0 = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5)
    face1 = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5)

    #containers for detected keypoints for each camera. These are filled at each frame.
    #This will run you into memory issue if you run the program without stop
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []
    while True:

        #read frames from stream
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1: break

        #crop to 720x720.
        #Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
        if frame0.shape[1] != 720:
            frame0 = frame0[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
            frame1 = frame1[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

        # the BGR image to RGB.
        frame0_rgb = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1_rgb = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame0_rgb.flags.writeable = False
        frame1_rgb.flags.writeable = False
        
        # Process all detections
        results0_pose = pose0.process(frame0_rgb)
        results1_pose = pose1.process(frame1_rgb)
        results0_hands = hands0.process(frame0_rgb)
        results1_hands = hands1.process(frame1_rgb)
        results0_face = face0.process(frame0_rgb)
        results1_face = face1.process(frame1_rgb)

        #reverse changes
        frame0_rgb.flags.writeable = True
        frame1_rgb.flags.writeable = True
        frame0 = cv.cvtColor(frame0_rgb, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1_rgb, cv.COLOR_RGB2BGR)

        #check for keypoints detection
        frame0_keypoints = []
        if results0_pose.pose_landmarks:
            for i, landmark in enumerate(results0_pose.pose_landmarks.landmark):
                if i not in pose_keypoints: continue #only save keypoints that are indicated in pose_keypoints
                pxl_x = landmark.x * frame0.shape[1]
                pxl_y = landmark.y * frame0.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                cv.circle(frame0,(pxl_x, pxl_y), 3, (0,0,255), -1) #add keypoint detection points into figure
                kpts = [pxl_x, pxl_y]
                frame0_keypoints.append(kpts)
        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame0_keypoints = [[-1, -1]]*len(pose_keypoints)

        #this will keep keypoints of this frame in memory
        kpts_cam0.append(frame0_keypoints)

        frame1_keypoints = []
        if results1_pose.pose_landmarks:
            for i, landmark in enumerate(results1_pose.pose_landmarks.landmark):
                if i not in pose_keypoints: continue
                pxl_x = landmark.x * frame1.shape[1]
                pxl_y = landmark.y * frame1.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                cv.circle(frame1,(pxl_x, pxl_y), 3, (0,0,255), -1)
                kpts = [pxl_x, pxl_y]
                frame1_keypoints.append(kpts)
        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame1_keypoints = [[-1, -1]]*len(pose_keypoints)

        #update keypoints container
        kpts_cam1.append(frame1_keypoints)

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

        # Draw face detections
        if results0_face.detections:
            for detection in results0_face.detections:
                mp_drawing.draw_detection(frame0, detection)
        
        if results1_face.detections:
            for detection in results1_face.detections:
                mp_drawing.draw_detection(frame1, detection)

        #calculate 3d position
        frame_p3ds = []
        for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                _p3d = DLT(P0, P1, uv1, uv2) #calculate 3d position of keypoint
            frame_p3ds.append(_p3d)

        '''
        This contains the 3d position of each keypoint in current frame.
        For real time application, this is what you want.
        '''
        frame_p3ds = np.array(frame_p3ds).reshape((12, 3))
        kpts_3d.append(frame_p3ds)

        cv.imshow('cam1', frame1)
        cv.imshow('cam0', frame0)

        k = cv.waitKey(1)
        if k & 0xFF == 27: break #27 is ESC key.

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)

if __name__ == '__main__':

    #this will load the sample videos if no camera ID is given
    input_stream1 = 'media/cam0_test.mp4'
    input_stream2 = 'media/cam1_test.mp4'

    #put camera id as command line arguements
    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])

    #get projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1)

    #this will create keypoints file in current working folder
    write_keypoints_to_disk('kpts_cam0.dat', kpts_cam0)
    write_keypoints_to_disk('kpts_cam1.dat', kpts_cam1)
    write_keypoints_to_disk('kpts_3d.dat', kpts_3d)
