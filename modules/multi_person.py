import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from datetime import datetime, timedelta

class VideoAnalysisRequest(BaseModel):
    video_path: str

app = FastAPI()

class MultiPersonDetector:
    def __init__(self):
        # Initialize YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # Using nano model for speed, can use larger models for better accuracy
        
        # Detection parameters
        self.SAMPLE_INTERVAL = 0.5  # Sample every 0.5 seconds
        self.CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections
        self.PERSON_CLASS_ID = 0  # COCO dataset class ID for person
        
    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """Detect persons in a frame using YOLOv8"""
        if frame is None:
            return []
            
        # Run YOLOv8 inference
        results = self.model(frame, conf=self.CONFIDENCE_THRESHOLD)[0]
        
        # Filter for person detections
        person_detections = []
        for box in results.boxes:
            if int(box.cls) == self.PERSON_CLASS_ID:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                person_detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence
                })
                
        return person_detections

def detect_multiple_persons(video_path: str) -> Dict:
    """Main function to detect multiple persons in a video"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    detector = MultiPersonDetector()
    
    # Initialize tracking variables
    total_frames_analyzed = 0
    frames_with_extra_people = 0
    first_extra_person_time = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * detector.SAMPLE_INTERVAL)
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Sample frames at specified interval
        if frame_count % frame_interval == 0:
            total_frames_analyzed += 1
            
            # Detect persons in frame
            detections = detector.detect_persons(frame)
            
            # Check for multiple persons
            if len(detections) > 1:
                frames_with_extra_people += 1
                
                # Record time of first extra person detection
                if first_extra_person_time is None:
                    first_extra_person_time = frame_count / fps
                    
        frame_count += 1
    
    cap.release()
    
    # Calculate suspicion score (0-100)
    score = min(100, (
        (frames_with_extra_people / max(total_frames_analyzed, 1) * 70) +  # Percentage of frames with extra people
        (30 if first_extra_person_time is not None else 0)  # Bonus for detecting extra people
    ))
    
    # Format time string
    time_str = "00:00:00"
    if first_extra_person_time is not None:
        time_str = str(timedelta(seconds=int(first_extra_person_time)))
    
    return {
        "total_frames_analyzed": total_frames_analyzed,
        "frames_with_extra_people": frames_with_extra_people,
        "first_extra_person_detected_at": time_str,
        "score": round(score, 1)
    }

@app.post("/multi_person")
async def analyze_video(request: VideoAnalysisRequest):
    try:
        results = detect_multiple_persons(request.video_path)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 