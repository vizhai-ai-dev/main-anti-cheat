from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DetectionMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float

@dataclass
class TimeMetrics:
    total_duration: float
    detection_duration: float
    processing_time: float
    frames_per_second: float

class MetricsCalculator:
    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []
    
    def calculate_detection_metrics(
        self,
        predictions: List[bool],
        ground_truth: List[bool]
    ) -> DetectionMetrics:
        """Calculate detection metrics"""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        true_positives = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
        false_positives = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
        false_negatives = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)) / len(predictions)
        
        return DetectionMetrics(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy
        )
    
    def calculate_time_metrics(
        self,
        total_duration: float,
        detection_duration: float,
        processing_time: float,
        frame_count: int
    ) -> TimeMetrics:
        """Calculate time-based metrics"""
        return TimeMetrics(
            total_duration=total_duration,
            detection_duration=detection_duration,
            processing_time=processing_time,
            frames_per_second=frame_count / processing_time if processing_time > 0 else 0
        )
    
    def calculate_risk_score(
        self,
        screen_switch_score: float,
        gaze_score: float,
        audio_score: float,
        multi_person_score: float,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, RiskLevel]:
        """Calculate risk score from individual module scores"""
        if weights is None:
            weights = {
                'screen_switch': 0.3,
                'gaze': 0.3,
                'audio': 0.2,
                'multi_person': 0.2
            }
        
        # Validate weights
        if not all(0 <= w <= 1 for w in weights.values()):
            raise ValueError("Weights must be between 0 and 1")
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1")
        
        # Calculate weighted score
        risk_score = (
            screen_switch_score * weights['screen_switch'] +
            gaze_score * weights['gaze'] +
            audio_score * weights['audio'] +
            multi_person_score * weights['multi_person']
        )
        
        # Determine risk level
        if risk_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.4:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return risk_score, risk_level
    
    def save_metrics(self, metrics: Dict[str, Any], filepath: str) -> None:
        """Save metrics to file"""
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_history.append(metrics)
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
    
    def load_metrics(self, filepath: str) -> List[Dict[str, Any]]:
        """Load metrics from file"""
        if not os.path.exists(filepath):
            return []
        
        with open(filepath, 'r') as f:
            self.metrics_history = json.load(f)
        
        return self.metrics_history
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Calculate average metrics from history"""
        if not self.metrics_history:
            return {}
        
        metrics_sum = {}
        count = len(self.metrics_history)
        
        for metrics in self.metrics_history:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metrics_sum[key] = metrics_sum.get(key, 0) + value
        
        return {key: value / count for key, value in metrics_sum.items()}
    
    def get_metrics_trend(
        self,
        metric_name: str,
        window_size: int = 10
    ) -> List[float]:
        """Get trend of a specific metric"""
        if not self.metrics_history:
            return []
        
        values = [m.get(metric_name, 0) for m in self.metrics_history]
        
        if len(values) <= window_size:
            return values
        
        return values[-window_size:]
    
    def plot_metrics_trend(
        self,
        metric_name: str,
        window_size: int = 10,
        save_path: Optional[str] = None
    ) -> None:
        """Plot trend of a specific metric"""
        import matplotlib.pyplot as plt
        
        values = self.get_metrics_trend(metric_name, window_size)
        if not values:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(values, marker='o')
        plt.title(f"{metric_name} Trend")
        plt.xlabel("Time")
        plt.ylabel(metric_name)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close() 