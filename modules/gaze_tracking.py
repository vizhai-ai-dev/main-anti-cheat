import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from datetime import datetime, timedelta

class VideoAnalysisRequest(BaseModel):
    video_path: str

app = FastAPI()

class GazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks indices for MediaPipe FaceMesh
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Gaze threshold parameters
        self.GAZE_THRESHOLD = 0.3  # Threshold for considering gaze as off-screen
        self.MIN_LOOK_AWAY_DURATION = 1.5  # Minimum duration (seconds) to flag as look away
        
    def get_eye_aspect_ratio(self, landmarks, eye_indices: List[int]) -> float:
        """Calculate the eye aspect ratio to determine if eye is open"""
        points = np.array([(landmarks[idx].x, landmarks[idx].y) for idx in eye_indices])
        return np.mean(points[:, 1])  # Average y-coordinate of eye landmarks
        
    def estimate_gaze_direction(self, landmarks, frame_shape: Tuple[int, int]) -> Tuple[float, float]:
        """Estimate gaze direction using eye landmarks"""
        # Get eye landmarks
        left_eye = np.array([(landmarks[idx].x, landmarks[idx].y) for idx in self.LEFT_EYE_INDICES])
        right_eye = np.array([(landmarks[idx].x, landmarks[idx].y) for idx in self.RIGHT_EYE_INDICES])
        
        # Calculate eye centers
        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)
        
        # Calculate gaze vector (simplified)
        gaze_x = (left_center[0] + right_center[0]) / 2 - 0.5  # Center normalized to [-0.5, 0.5]
        gaze_y = (left_center[1] + right_center[1]) / 2 - 0.5
        
        return gaze_x, gaze_y
        
    def is_looking_away(self, gaze_x: float, gaze_y: float) -> bool:
        """Determine if the gaze is directed away from the screen"""
        return abs(gaze_x) > self.GAZE_THRESHOLD or abs(gaze_y) > self.GAZE_THRESHOLD

def detect_gaze_deviation(video_path: str) -> Dict:
    """Main function to detect gaze deviations in a video"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    tracker = GazeTracker()
    
    # Initialize counters and tracking variables
    frames_tracked = 0
    frames_looked_away = 0
    current_look_away_duration = 0
    max_look_away_duration = 0
    look_away_segments = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = tracker.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            frames_tracked += 1
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Estimate gaze direction
            gaze_x, gaze_y = tracker.estimate_gaze_direction(landmarks, frame.shape[:2])
            
            # Check if looking away
            if tracker.is_looking_away(gaze_x, gaze_y):
                frames_looked_away += 1
                current_look_away_duration += 1/fps
                max_look_away_duration = max(max_look_away_duration, current_look_away_duration)
            else:
                if current_look_away_duration > 0:
                    look_away_segments.append(current_look_away_duration)
                current_look_away_duration = 0
    
    cap.release()
    
    # Calculate if there's recurrent off-screen behavior
    recurrent_offscreen = False
    if look_away_segments:
        # Consider it recurrent if there are multiple segments longer than threshold
        long_segments = [seg for seg in look_away_segments if seg > tracker.MIN_LOOK_AWAY_DURATION]
        recurrent_offscreen = len(long_segments) >= 2
    
    # Calculate suspicion score (0-100)
    score = min(100, (
        (frames_looked_away / max(frames_tracked, 1) * 50) +  # Percentage of frames looking away
        (max_look_away_duration * 10) +  # Duration of longest look away
        (20 if recurrent_offscreen else 0)  # Bonus for recurrent behavior
    ))
    
    return {
        "frames_tracked": frames_tracked,
        "frames_looked_away": frames_looked_away,
        "max_duration_look_away": round(max_look_away_duration, 1),
        "recurrent_offscreen_behavior": recurrent_offscreen,
        "score": round(score, 1)
    }

@app.post("/gaze_tracking")
async def analyze_video(request: VideoAnalysisRequest):
    try:
        results = detect_gaze_deviation(request.video_path)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 