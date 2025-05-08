import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

def draw_bounding_box(
    image: np.ndarray,
    box: List[float],
    label: Optional[str] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """Draw bounding box on image"""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 4),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return image

def draw_gaze_direction(
    image: np.ndarray,
    eye_center: Tuple[int, int],
    gaze_vector: Tuple[float, float],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    length: int = 50
) -> np.ndarray:
    """Draw gaze direction on image"""
    x, y = eye_center
    dx, dy = gaze_vector
    
    # Calculate end point
    end_x = int(x + dx * length)
    end_y = int(y + dy * length)
    
    # Draw line
    cv2.line(image, (x, y), (end_x, end_y), color, thickness)
    
    # Draw eye center
    cv2.circle(image, (x, y), 3, color, -1)
    
    return image

def draw_timeline(
    events: List[Dict[str, Any]],
    duration: float,
    width: int = 800,
    height: int = 100,
    save_path: Optional[str] = None
) -> None:
    """Draw timeline of events"""
    plt.figure(figsize=(width/100, height/100))
    
    # Create timeline
    plt.plot([0, duration], [0, 0], 'k-', linewidth=2)
    
    # Plot events
    colors = {'screen_switch': 'red', 'gaze': 'blue', 'audio': 'green', 'multi_person': 'purple'}
    
    for event in events:
        event_type = event['type']
        timestamp = event['timestamp']
        confidence = event.get('confidence', 1.0)
        
        plt.plot(
            timestamp,
            0,
            'o',
            color=colors.get(event_type, 'gray'),
            markersize=10 * confidence,
            alpha=0.7
        )
    
    # Customize plot
    plt.title('Detection Timeline')
    plt.xlabel('Time (seconds)')
    plt.yticks([])
    plt.grid(True, axis='x')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=event_type, markersize=10)
        for event_type, color in colors.items()
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def create_heatmap(
    detections: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    sigma: float = 20.0,
    save_path: Optional[str] = None
) -> np.ndarray:
    """Create heatmap from detections"""
    heatmap = np.zeros(image_shape[:2], dtype=np.float32)
    
    for detection in detections:
        if 'bounding_box' in detection:
            x1, y1, x2, y2 = map(int, detection['bounding_box'])
            confidence = detection.get('confidence', 1.0)
            
            # Create Gaussian kernel
            kernel_size = int(sigma * 6)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            kernel = cv2.getGaussianKernel(kernel_size, sigma)
            kernel_2d = kernel * kernel.T
            
            # Add to heatmap
            heatmap[y1:y2, x1:x2] += kernel_2d[:y2-y1, :x2-x1] * confidence
    
    # Normalize heatmap
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    if save_path:
        cv2.imwrite(save_path, heatmap)
    
    return heatmap

def create_summary_video(
    video_path: str,
    detections: List[Dict[str, Any]],
    output_path: str,
    fps: float = 30.0
) -> None:
    """Create summary video with detections"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get detections for current frame
        current_time = frame_count / fps
        frame_detections = [
            d for d in detections
            if abs(d['timestamp'] - current_time) < 1.0/fps
        ]
        
        # Draw detections
        for detection in frame_detections:
            if 'bounding_box' in detection:
                frame = draw_bounding_box(
                    frame,
                    detection['bounding_box'],
                    f"{detection['type']}: {detection.get('confidence', 1.0):.2f}"
                )
            
            if 'gaze_vector' in detection:
                frame = draw_gaze_direction(
                    frame,
                    detection['eye_center'],
                    detection['gaze_vector']
                )
        
        # Write frame
        out.write(frame)
        frame_count += 1
    
    # Release resources
    cap.release()
    out.release()

def plot_metrics(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """Plot multiple metrics"""
    plt.figure(figsize=(12, 6))
    
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name, marker='o')
    
    plt.title('Detection Metrics')
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def create_detection_report(
    detections: List[Dict[str, Any]],
    output_dir: str
) -> None:
    """Create comprehensive detection report"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timeline
    timeline_path = os.path.join(output_dir, 'timeline.png')
    draw_timeline(detections, max(d['timestamp'] for d in detections), save_path=timeline_path)
    
    # Create heatmap
    if any('bounding_box' in d for d in detections):
        heatmap_path = os.path.join(output_dir, 'heatmap.png')
        create_heatmap(detections, (720, 1280), save_path=heatmap_path)
    
    # Create metrics plot
    metrics = {
        'screen_switch': [d.get('confidence', 0) for d in detections if d['type'] == 'screen_switch'],
        'gaze': [d.get('confidence', 0) for d in detections if d['type'] == 'gaze'],
        'audio': [d.get('confidence', 0) for d in detections if d['type'] == 'audio'],
        'multi_person': [d.get('confidence', 0) for d in detections if d['type'] == 'multi_person']
    }
    metrics_path = os.path.join(output_dir, 'metrics.png')
    plot_metrics(metrics, save_path=metrics_path)
    
    # Save detections as JSON
    json_path = os.path.join(output_dir, 'detections.json')
    with open(json_path, 'w') as f:
        json.dump(detections, f, indent=4) 