from typing import Dict, List, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class ModuleResults(BaseModel):
    screen_switch: Dict
    gaze: Dict
    audio: Dict
    multi_person: Dict

app = FastAPI()

class CheatScoreCalculator:
    def __init__(self):
        # Module weights for final score
        self.MODULE_WEIGHTS = {
            'screen_switch': 0.25,
            'gaze': 0.25,
            'audio': 0.25,
            'multi_person': 0.25
        }
        
        # Risk level thresholds
        self.RISK_THRESHOLDS = {
            RiskLevel.LOW: 30,
            RiskLevel.MEDIUM: 50,
            RiskLevel.HIGH: 70,
            RiskLevel.CRITICAL: 90
        }
        
    def get_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level based on score"""
        if score >= self.RISK_THRESHOLDS[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif score >= self.RISK_THRESHOLDS[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif score >= self.RISK_THRESHOLDS[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
        
    def get_screen_switch_reasons(self, results: Dict) -> List[str]:
        """Extract reasons from screen switch results"""
        reasons = []
        
        if results.get('fullscreen_violations', 0) > 0:
            reasons.append(f"Fullscreen violated {results['fullscreen_violations']} times")
            
        if results.get('overlay_detected', False):
            reasons.append("Overlay detected")
            
        if results.get('scene_switch_events', 0) > 5:
            reasons.append(f"Multiple scene switches detected ({results['scene_switch_events']})")
            
        return reasons
        
    def get_gaze_reasons(self, results: Dict) -> List[str]:
        """Extract reasons from gaze tracking results"""
        reasons = []
        
        if results.get('frames_looked_away', 0) > 0:
            reasons.append(f"Looked away from screen {results['frames_looked_away']} times")
            
        if results.get('recurrent_offscreen_behavior', False):
            reasons.append("Recurrent off-screen behavior detected")
            
        if results.get('max_duration_look_away', 0) > 2.0:
            reasons.append(f"Long look-away duration: {results['max_duration_look_away']}s")
            
        return reasons
        
    def get_audio_reasons(self, results: Dict) -> List[str]:
        """Extract reasons from audio analysis results"""
        reasons = []
        
        if results.get('num_speakers', 0) > 1:
            reasons.append(f"Multiple speakers detected ({results['num_speakers']})")
            
        if results.get('whispers_detected', False):
            reasons.append("Whispers detected")
            
        if results.get('whisper_segments', []):
            reasons.append(f"Multiple whisper segments ({len(results['whisper_segments'])})")
            
        return reasons
        
    def get_multi_person_reasons(self, results: Dict) -> List[str]:
        """Extract reasons from multi-person detection results"""
        reasons = []
        
        if results.get('frames_with_extra_people', 0) > 0:
            reasons.append(f"Extra people detected in {results['frames_with_extra_people']} frames")
            
        if results.get('first_extra_person_detected_at', "00:00:00") != "00:00:00":
            reasons.append(f"First extra person detected at {results['first_extra_person_detected_at']}")
            
        return reasons

def calculate_risk_score(
    screen_results: Optional[Dict[str, Any]] = None,
    gaze_results: Optional[Dict[str, Any]] = None,
    person_results: Optional[Dict[str, Any]] = None,
    audio_results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate overall risk score based on results from all detection modules.
    
    Args:
        screen_results: Results from screen change detection
        gaze_results: Results from gaze tracking
        person_results: Results from person detection
        audio_results: Results from audio analysis
        
    Returns:
        Dict containing:
            - risk_score: Overall risk score (0-100)
            - risk_level: Low/Medium/High
            - reasons: List of reasons for the risk assessment
            - module_scores: Individual scores from each module
    """
    
    # Initialize results
    results = {
        'risk_score': 0,
        'risk_level': 'Low',
        'reasons': [],
        'module_scores': {}
    }
    
    try:
        # Screen monitoring score (30% weight)
        screen_score = 0
        if screen_results:
            screen_violations = (
                screen_results.get('fullscreen_violations', 0) +
                screen_results.get('overlay_detections', 0) +
                screen_results.get('scene_switches', 0)
            )
            screen_score = min(100, screen_violations * 20)  # 20 points per violation
            results['module_scores']['screen'] = screen_score
            
            if screen_violations > 0:
                results['reasons'].append(
                    f"Detected {screen_violations} screen violations"
                )
                
        # Gaze tracking score (25% weight)
        gaze_score = 0
        if gaze_results:
            off_screen_ratio = gaze_results.get('off_screen_ratio', 0)
            rapid_movements = gaze_results.get('rapid_movements', 0)
            
            gaze_score = min(100, (
                off_screen_ratio * 70 +  # Up to 70 points for looking away
                rapid_movements * 10     # 10 points per suspicious eye movement
            ))
            results['module_scores']['gaze'] = gaze_score
            
            if off_screen_ratio > 0.2:  # More than 20% time looking away
                results['reasons'].append(
                    f"Subject looked away from screen {int(off_screen_ratio * 100)}% of the time"
                )
                
        # Person detection score (25% weight)
        person_score = 0
        if person_results:
            extra_people = person_results.get('max_people_detected', 1) - 1
            person_score = min(100, extra_people * 100)  # 100 points per extra person
            results['module_scores']['person'] = person_score
            
            if extra_people > 0:
                results['reasons'].append(
                    f"Detected {extra_people} additional people in frame"
                )
                
        # Audio analysis score (20% weight)
        audio_score = 0
        if audio_results:
            if audio_results.get('has_audio', True):  # Only consider if video has audio
                audio_score = audio_results.get('score', 0)
                results['module_scores']['audio'] = audio_score
                
                if audio_results.get('num_speakers', 1) > 1:
                    results['reasons'].append(
                        f"Detected {audio_results['num_speakers']} different speakers"
                    )
                if audio_results.get('whispers_detected', False):
                    results['reasons'].append("Detected whispered speech")
                    
        # Calculate weighted average
        weights = {
            'screen': 0.30,
            'gaze': 0.25,
            'person': 0.25,
            'audio': 0.20
        }
        
        total_weight = 0
        weighted_score = 0
        
        for module, score in results['module_scores'].items():
            weighted_score += score * weights[module]
            total_weight += weights[module]
            
        if total_weight > 0:
            results['risk_score'] = round(weighted_score / total_weight, 1)
            
        # Determine risk level
        if results['risk_score'] >= 70:
            results['risk_level'] = 'High'
        elif results['risk_score'] >= 40:
            results['risk_level'] = 'Medium'
        else:
            results['risk_level'] = 'Low'
            
        # Add summary reason if no specific reasons found
        if not results['reasons']:
            if results['risk_score'] < 20:
                results['reasons'].append("No suspicious behavior detected")
            else:
                results['reasons'].append("Multiple minor suspicious indicators detected")
                
    except Exception as e:
        logger.error(f"Error calculating risk score: {str(e)}")
        raise
        
    return results

@app.post("/cheat_score")
async def calculate_score(results: ModuleResults):
    try:
        score_results = calculate_risk_score(
            screen_results=results.screen_switch,
            gaze_results=results.gaze,
            person_results=results.multi_person,
            audio_results=results.audio
        )
        return score_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 