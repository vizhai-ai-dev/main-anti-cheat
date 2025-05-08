import cv2
import numpy as np
from typing import Tuple, List, Optional
import os
from datetime import datetime

def extract_frames(video_path: str, sample_interval: float = 0.5) -> List[Tuple[float, np.ndarray]]:
    """Extract frames from video at specified intervals"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * sample_interval)
    
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            frames.append((timestamp, frame))
            
        frame_count += 1
        
    cap.release()
    return frames

def get_video_info(video_path: str) -> dict:
    """Get video metadata"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info

def save_frame(frame: np.ndarray, output_dir: str, timestamp: float) -> str:
    """Save a frame to disk with timestamp"""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"frame_{timestamp:.2f}.jpg"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, frame)
    return output_path

def create_video_from_frames(frames: List[np.ndarray], output_path: str, fps: float = 30) -> str:
    """Create a video from a list of frames"""
    if not frames:
        raise ValueError("No frames provided")
        
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
        
    out.release()
    return output_path

def detect_scene_changes(frames: List[np.ndarray], threshold: float = 30.0) -> List[int]:
    """Detect scene changes in a sequence of frames"""
    if len(frames) < 2:
        return []
        
    changes = []
    prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    for i, frame in enumerate(frames[1:], 1):
        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram difference
        hist1 = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([curr_frame], [0], None, [256], [0, 256])
        
        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        if diff < threshold:
            changes.append(i)
            
        prev_frame = curr_frame
        
    return changes 