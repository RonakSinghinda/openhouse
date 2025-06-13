# Multi-Exercise Fitness Tracker

A real-time exercise detection and counting system using MediaPipe and Python.

## Features
- Real-time exercise detection and counting
- Supports multiple exercises:
  - Squats
  - Lunges
  - Push-ups
  - Jumping jacks
- Visual feedback with exercise count and form guidance
- Pose estimation using MediaPipe

## Setup
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python exercise_tracker.py
```

## Usage
- Press 'q' to quit the application
- The system will automatically detect and count exercises
- Current exercise type and count will be displayed on screen

## Requirements
- Python 3.8+
- Webcam
- See requirements.txt for Python package dependencies 