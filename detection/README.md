# Body Pose Detection

A Python-based body pose detection system that uses MediaPipe to track and analyze human body movements in real-time.

## Features

- Real-time body pose tracking
- Detection of body position (Standing Straight, Leaning Left/Right)
- Shoulder and hip angle calculations
- Hand finger counting
- Visual feedback with skeleton overlay

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the program:
```bash
python body_pose_detection.py
```

2. Controls:
- Press 'q' to quit the program

3. Features:
- Body skeleton overlay
- Position detection (Standing Straight, Leaning Left/Right)
- Shoulder and hip angle display
- Hand finger counting

## How it Works

The system uses MediaPipe's Pose and Hands modules to:
1. Detect and track 33 body landmarks
2. Calculate body position based on shoulder and hip angles
3. Track hand positions and count extended fingers
4. Provide real-time visual feedback

## Contributing

Feel free to submit issues and enhancement requests! 