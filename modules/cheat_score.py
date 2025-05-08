from typing import Dict, List, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from enum import Enum

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

def compute_cheating_score(results_dict: Dict) -> Dict:
    """Compute final cheating score from all module results"""
    calculator = CheatScoreCalculator()
    
    # Extract module scores
    screen_score = results_dict['screen_switch'].get('suspicion_score', 0)
    gaze_score = results_dict['gaze'].get('score', 0)
    audio_score = results_dict['audio'].get('score', 0)
    multi_score = results_dict['multi_person'].get('score', 0)
    
    # Calculate weighted final score
    final_score = (
        screen_score * calculator.MODULE_WEIGHTS['screen_switch'] +
        gaze_score * calculator.MODULE_WEIGHTS['gaze'] +
        audio_score * calculator.MODULE_WEIGHTS['audio'] +
        multi_score * calculator.MODULE_WEIGHTS['multi_person']
    )
    
    # Determine risk level
    risk_level = calculator.get_risk_level(final_score)
    
    # Collect reasons from all modules
    reasons = []
    reasons.extend(calculator.get_screen_switch_reasons(results_dict['screen_switch']))
    reasons.extend(calculator.get_gaze_reasons(results_dict['gaze']))
    reasons.extend(calculator.get_audio_reasons(results_dict['audio']))
    reasons.extend(calculator.get_multi_person_reasons(results_dict['multi_person']))
    
    return {
        "final_score": round(final_score, 1),
        "risk": risk_level,
        "reasons": reasons,
        "module_scores": {
            "screen_switch": round(screen_score, 1),
            "gaze": round(gaze_score, 1),
            "audio": round(audio_score, 1),
            "multi_person": round(multi_score, 1)
        }
    }

@app.post("/cheat_score")
async def calculate_score(results: ModuleResults):
    try:
        score_results = compute_cheating_score(results.dict())
        return score_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 