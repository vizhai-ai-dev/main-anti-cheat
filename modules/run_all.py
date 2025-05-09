import os
import asyncio
import subprocess
import json
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
from datetime import datetime
import logging
import aiohttp
import importlib.util
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoAnalysisRequest(BaseModel):
    video_path: str

app = FastAPI()

def run_all_modules(video_path: str) -> Dict[str, Any]:
    """
    Run all detection modules on the video and return aggregated results.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        Dict[str, Any]: Aggregated results from all modules
    """
    try:
        # Verify video exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Get the modules directory path
        modules_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Import modules using absolute paths
        screen_switch_path = os.path.join(modules_dir, "screen_switch.py")
        gaze_tracking_path = os.path.join(modules_dir, "gaze_tracking.py")
        audio_analysis_path = os.path.join(modules_dir, "audio_analysis.py")
        multi_person_path = os.path.join(modules_dir, "multi_person.py")
        cheat_score_path = os.path.join(modules_dir, "cheat_score.py")
        
        # Import modules
        screen_switch_spec = importlib.util.spec_from_file_location("screen_switch", screen_switch_path)
        screen_switch = importlib.util.module_from_spec(screen_switch_spec)
        screen_switch_spec.loader.exec_module(screen_switch)
        
        gaze_spec = importlib.util.spec_from_file_location("gaze_tracking", gaze_tracking_path)
        gaze = importlib.util.module_from_spec(gaze_spec)
        gaze_spec.loader.exec_module(gaze)
        
        audio_spec = importlib.util.spec_from_file_location("audio_analysis", audio_analysis_path)
        audio = importlib.util.module_from_spec(audio_spec)
        audio_spec.loader.exec_module(audio)
        
        multi_person_spec = importlib.util.spec_from_file_location("multi_person", multi_person_path)
        multi_person = importlib.util.module_from_spec(multi_person_spec)
        multi_person_spec.loader.exec_module(multi_person)
        
        cheat_score_spec = importlib.util.spec_from_file_location("cheat_score", cheat_score_path)
        cheat_score = importlib.util.module_from_spec(cheat_score_spec)
        cheat_score_spec.loader.exec_module(cheat_score)
        
        # Run all modules
        results = {}
        
        # Screen switch detection
        screen_results = screen_switch.detect_screen_anomalies(video_path)
        results['screen_switch'] = screen_results
        
        # Gaze tracking
        gaze_results = gaze.detect_gaze_deviation(video_path)
        results['gaze'] = gaze_results
        
        # Audio analysis
        audio_results = audio.detect_audio_anomalies(video_path)
        results['audio'] = audio_results
        
        # Multi-person detection
        multi_person_results = multi_person.detect_multiple_persons(video_path)
        results['multi_person'] = multi_person_results
        
        # Calculate final risk score
        risk_score = cheat_score.calculate_risk_score(results)
        results['risk_score'] = risk_score
        
        # Determine risk level
        if risk_score >= 80:
            results['risk_level'] = 'HIGH'
        elif risk_score >= 50:
            results['risk_level'] = 'MEDIUM'
        else:
            results['risk_level'] = 'LOW'
            
        return results
        
    except Exception as e:
        logger.error(f"Error running modules: {str(e)}")
        raise

class ModuleOrchestrator:
    def __init__(self):
        self.modules = {
            'screen_switch': 'screen_switch.py',
            'gaze': 'gaze_tracking.py',
            'audio': 'audio_analysis.py',
            'multi_person': 'multi_person.py'
        }
        self.cheat_score_module = 'cheat_score.py'
        
    async def run_module(self, module_name: str, video_path: str) -> Dict:
        """Run a single module asynchronously"""
        try:
            # Start the module as a subprocess
            process = await asyncio.create_subprocess_exec(
                'python', self.modules[module_name],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await asyncio.sleep(2)  # Give time for FastAPI to start
            
            # Make the API call to the module
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'http://localhost:8000/{module_name}',
                    json={'video_path': video_path}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"{module_name} analysis completed successfully")
                        return result
                    else:
                        error = await response.text()
                        logger.error(f"Error in {module_name}: {error}")
                        return {}
                        
        except Exception as e:
            logger.error(f"Error running {module_name}: {str(e)}")
            return {}
            
    async def run_all_modules_async(self, video_path: str) -> Dict:
        """Run all modules in parallel and aggregate results"""
        # Verify video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Start all modules in parallel
        tasks = []
        for module_name in self.modules:
            tasks.append(self.run_module(module_name, video_path))
            
        # Wait for all modules to complete
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        aggregated_results = {
            'screen_switch': results[0],
            'gaze': results[1],
            'audio': results[2],
            'multi_person': results[3]
        }
        
        # Calculate final cheat score
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'http://localhost:8000/cheat_score',
                    json=aggregated_results
                ) as response:
                    if response.status == 200:
                        final_score = await response.json()
                        aggregated_results['final_analysis'] = final_score
                    else:
                        error = await response.text()
                        logger.error(f"Error calculating final score: {error}")
                        
        except Exception as e:
            logger.error(f"Error in final score calculation: {str(e)}")
            
        return aggregated_results

@app.post("/run_all")
async def analyze_video(request: VideoAnalysisRequest, background_tasks: BackgroundTasks):
    """Endpoint to run all analysis modules"""
    try:
        orchestrator = ModuleOrchestrator()
        results = await orchestrator.run_all_modules_async(request.video_path)
        
        # Add metadata
        results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'video_path': request.video_path,
            'analysis_duration': 'N/A'  # Could be calculated if needed
        }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def cleanup():
    """Cleanup function to stop all module servers"""
    try:
        # Find and kill all Python processes running our modules
        for module in ModuleOrchestrator().modules.values():
            subprocess.run(['pkill', '-f', module])
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup when the main server shuts down"""
    cleanup()

if __name__ == "__main__":
    # Test the module
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        results = run_all_modules(video_path)
        print(json.dumps(results, indent=2))
    else:
        try:
            uvicorn.run(app, host="0.0.0.0", port=8000)
        finally:
            cleanup()