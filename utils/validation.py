from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
import numpy as np
from datetime import datetime
import json

# Base validation models
class TimeRange(BaseModel):
    start: float
    end: float
    
    @validator('end')
    def end_must_be_after_start(cls, v, values):
        if 'start' in values and v <= values['start']:
            raise ValueError('end time must be after start time')
        return v

class DetectionResult(BaseModel):
    timestamp: float
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

class VideoMetadata(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0.0)
    duration: float = Field(gt=0.0)
    frame_count: int = Field(gt=0)
    format: str

class AudioMetadata(BaseModel):
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    duration: float = Field(gt=0.0)
    format: str

# Validation functions
def validate_video_path(video_path: str) -> None:
    """Validate video file path"""
    if not video_path:
        raise ValueError("Video path cannot be empty")
    
    if not video_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise ValueError("Invalid video file format")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

def validate_audio_path(audio_path: str) -> None:
    """Validate audio file path"""
    if not audio_path:
        raise ValueError("Audio path cannot be empty")
    
    if not audio_path.endswith(('.wav', '.mp3', '.m4a')):
        raise ValueError("Invalid audio file format")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

def validate_timestamp(timestamp: float) -> None:
    """Validate timestamp"""
    if not isinstance(timestamp, (int, float)):
        raise TypeError("Timestamp must be a number")
    
    if timestamp < 0:
        raise ValueError("Timestamp cannot be negative")

def validate_confidence(confidence: float) -> None:
    """Validate confidence score"""
    if not isinstance(confidence, (int, float)):
        raise TypeError("Confidence must be a number")
    
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1")

def validate_bounding_box(box: List[float]) -> None:
    """Validate bounding box coordinates"""
    if not isinstance(box, list):
        raise TypeError("Bounding box must be a list")
    
    if len(box) != 4:
        raise ValueError("Bounding box must have 4 coordinates")
    
    for coord in box:
        if not isinstance(coord, (int, float)):
            raise TypeError("Bounding box coordinates must be numbers")
        if coord < 0:
            raise ValueError("Bounding box coordinates cannot be negative")

# Data processing functions
def normalize_confidence(confidence: float) -> float:
    """Normalize confidence score to [0, 1] range"""
    return max(0.0, min(1.0, confidence))

def normalize_bounding_box(
    box: List[float],
    image_width: int,
    image_height: int
) -> List[float]:
    """Normalize bounding box coordinates to [0, 1] range"""
    x1, y1, x2, y2 = box
    return [
        x1 / image_width,
        y1 / image_height,
        x2 / image_width,
        y2 / image_height
    ]

def denormalize_bounding_box(
    box: List[float],
    image_width: int,
    image_height: int
) -> List[float]:
    """Convert normalized bounding box coordinates to pixel values"""
    x1, y1, x2, y2 = box
    return [
        x1 * image_width,
        y1 * image_height,
        x2 * image_width,
        y2 * image_height
    ]

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def merge_overlapping_boxes(
    boxes: List[List[float]],
    iou_threshold: float = 0.5
) -> List[List[float]]:
    """Merge overlapping bounding boxes"""
    if not boxes:
        return []
    
    # Sort boxes by area (largest first)
    boxes = sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
    merged = []
    
    while boxes:
        current = boxes.pop(0)
        overlapping = []
        
        for box in boxes[:]:
            if calculate_iou(current, box) > iou_threshold:
                overlapping.append(box)
                boxes.remove(box)
        
        if overlapping:
            # Merge overlapping boxes
            x1 = min(current[0], *[box[0] for box in overlapping])
            y1 = min(current[1], *[box[1] for box in overlapping])
            x2 = max(current[2], *[box[2] for box in overlapping])
            y2 = max(current[3], *[box[3] for box in overlapping])
            merged.append([x1, y1, x2, y2])
        else:
            merged.append(current)
    
    return merged

def filter_detections(
    detections: List[DetectionResult],
    min_confidence: float = 0.5,
    max_detections: Optional[int] = None
) -> List[DetectionResult]:
    """Filter detections based on confidence and limit"""
    filtered = [d for d in detections if d.confidence >= min_confidence]
    
    if max_detections is not None:
        filtered = sorted(filtered, key=lambda x: x.confidence, reverse=True)[:max_detections]
    
    return filtered 