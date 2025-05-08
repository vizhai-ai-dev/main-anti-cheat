import cv2
import numpy as np
import pytesseract
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from skimage.metrics import structural_similarity as ssim

class VideoAnalysisRequest(BaseModel):
    video_path: str

app = FastAPI()

class ScreenAnomalyDetector:
    def __init__(self):
        self.scene_threshold = 30.0  # Threshold for scene change detection
        self.overlay_templates = self._load_overlay_templates()
        self.edge_threshold = 0.95  # Threshold for fullscreen detection
        
    def _load_overlay_templates(self) -> Dict[str, np.ndarray]:
        """Load template images for common overlays (taskbar, dock, etc.)"""
        templates = {}
        # TODO: Add actual template images for different OS overlays
        return templates

    def detect_scene_changes(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Detect significant scene changes using histogram comparison"""
        if frame1 is None or frame2 is None:
            return False
            
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate histograms
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # Compare histograms
        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return diff < self.scene_threshold

    def detect_overlays(self, frame: np.ndarray) -> bool:
        """Detect UI overlays using template matching and edge detection"""
        if frame is None:
            return False
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        edges = np.uint8(np.absolute(edges))
        
        # Check for horizontal lines (common in UI overlays)
        horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        # Template matching for known overlays
        overlay_detected = False
        for template_name, template in self.overlay_templates.items():
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            if np.max(result) > 0.8:  # Threshold for template matching
                overlay_detected = True
                break
                
        return overlay_detected or (horizontal_lines is not None and len(horizontal_lines) > 0)

    def detect_fullscreen_violation(self, frame: np.ndarray) -> bool:
        """Detect if the content is not in fullscreen mode"""
        if frame is None:
            return False
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check edges for consistent margins
        height, width = gray.shape
        top_edge = gray[0:5, :]
        bottom_edge = gray[-5:, :]
        
        # Calculate edge consistency
        top_consistency = np.std(top_edge) < 10
        bottom_consistency = np.std(bottom_edge) < 10
        
        return top_consistency or bottom_consistency

    def detect_cursor_and_ocr(self, frame: np.ndarray) -> Tuple[bool, bool]:
        """Detect cursor motion and perform OCR for text detection"""
        if frame is None:
            return False, False
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Cursor detection (simplified - looking for small bright regions)
        cursor_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
        cursor_detected = np.sum(cursor_mask) > 1000
        
        # OCR for text detection
        text = pytesseract.image_to_string(gray)
        text_detected = len(text.strip()) > 0
        
        return cursor_detected, text_detected

def detect_screen_anomalies(video_path: str) -> Dict:
    """Main function to detect screen anomalies in a video"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    detector = ScreenAnomalyDetector()
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Default to 30 FPS if not available
    
    # Initialize counters and flags
    fullscreen_violations = 0
    overlay_detected = False
    scene_switch_events = 0
    anomaly_start_time = None
    total_frames = 0
    
    prev_frame = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        total_frames += 1
        
        # Detect scene changes
        if prev_frame is not None and detector.detect_scene_changes(prev_frame, frame):
            scene_switch_events += 1
            
        # Detect overlays
        if detector.detect_overlays(frame):
            overlay_detected = True
            if anomaly_start_time is None:
                anomaly_start_time = total_frames / fps
                
        # Detect fullscreen violations
        if detector.detect_fullscreen_violation(frame):
            fullscreen_violations += 1
            if anomaly_start_time is None:
                anomaly_start_time = total_frames / fps
                
        prev_frame = frame.copy()
        
    cap.release()
    
    # Calculate duration with anomalies
    duration_with_anomalies = "00:00:00"
    if anomaly_start_time is not None and total_frames > 0:
        duration = timedelta(seconds=int(total_frames / fps - anomaly_start_time))
        duration_with_anomalies = str(duration)
    
    # Calculate suspicion score (0-100)
    suspicion_score = min(100, (
        (fullscreen_violations * 10) +
        (scene_switch_events * 5) +
        (50 if overlay_detected else 0)
    ))
    
    return {
        "fullscreen_violations": fullscreen_violations,
        "overlay_detected": overlay_detected,
        "scene_switch_events": scene_switch_events,
        "duration_with_anomalies": duration_with_anomalies,
        "suspicion_score": suspicion_score
    }

@app.post("/screen_switch")
async def analyze_video(request: VideoAnalysisRequest):
    try:
        results = detect_screen_anomalies(request.video_path)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 