import os
import ffmpeg
import whisper
import torch
import numpy as np
from pyannote.audio import Pipeline
from typing import Dict, List, Tuple, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import json
import tempfile
import cv2
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoAnalysisRequest(BaseModel):
    video_path: str

app = FastAPI()

class AudioAnalyzer:
    def __init__(self):
        # Initialize Whisper model
        self.whisper_model = whisper.load_model("base")
        
        # Initialize pyannote.audio pipeline
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=os.getenv("HF_TOKEN")  # HuggingFace token required
        )
        
        # Initialize Silero VAD
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True
        )
        self.get_speech_timestamps, _, self.read_audio, *_ = utils
        
        # Analysis parameters
        self.WHISPER_THRESHOLD = -20  # dB threshold for whisper detection
        self.MIN_SPEECH_DURATION = 0.5  # Minimum duration for speech segment
        
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video using ffmpeg"""
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio.close()
        
        try:
            # Use ffmpeg-python to extract audio
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream, temp_audio.name, acodec='pcm_s16le', ac=1, ar='16k')
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)
            return temp_audio.name
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise Exception(f"Error extracting audio: {str(e)}")
        except Exception as e:
            logger.error(f"Error in extract_audio: {str(e)}")
            raise
            
    def detect_whispers(self, audio_path: str) -> List[Dict]:
        """Detect whisper-like segments using audio analysis"""
        # Load audio
        audio = self.read_audio(audio_path, sampling_rate=16000)
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio, 
            self.vad_model,
            threshold=0.5,
            sampling_rate=16000
        )
        
        # Analyze each speech segment for whisper characteristics
        whisper_segments = []
        for ts in speech_timestamps:
            segment = audio[ts['start']:ts['end']]
            # Calculate energy and spectral characteristics
            energy = np.mean(np.abs(segment))
            if energy < self.WHISPER_THRESHOLD:
                whisper_segments.append({
                    'start': ts['start'] / 16000,  # Convert to seconds
                    'end': ts['end'] / 16000
                })
                
        return whisper_segments
        
    def analyze_speakers(self, audio_path: str) -> Tuple[int, List[Dict]]:
        """Analyze speaker diarization using pyannote.audio"""
        # Run diarization
        diarization = self.diarization_pipeline(audio_path)
        
        # Extract unique speakers and their segments
        speakers = set()
        speaker_segments = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            speaker_segments.append({
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end
            })
            
        return len(speakers), speaker_segments

def analyze_audio(video_path: str) -> Dict:
    """Main function to analyze audio in a video"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    analyzer = AudioAnalyzer()
    
    try:
        # Extract audio from video
        audio_path = analyzer.extract_audio(video_path)
        
        # Get video duration
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        
        # Detect whispers
        whisper_segments = analyzer.detect_whispers(audio_path)
        whispers_detected = len(whisper_segments) > 0
        
        # Analyze speakers
        num_speakers, speaker_segments = analyzer.analyze_speakers(audio_path)
        
        # Calculate suspicion score (0-100)
        score = min(100, (
            (len(whisper_segments) * 10) +  # Points for each whisper segment
            ((num_speakers - 1) * 20) +  # Points for additional speakers
            (30 if whispers_detected else 0)  # Bonus for detecting whispers
        ))
        
        # Format duration
        duration_str = str(timedelta(seconds=int(duration)))
        
        return {
            "total_duration": duration_str,
            "num_speakers": num_speakers,
            "whispers_detected": whispers_detected,
            "whisper_segments": whisper_segments,
            "score": round(score, 1)
        }
        
    finally:
        # Clean up temporary audio file
        if 'audio_path' in locals():
            os.unlink(audio_path)

@app.post("/audio_analysis")
async def analyze_video(request: VideoAnalysisRequest):
    try:
        results = analyze_audio(request.video_path)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def detect_audio_anomalies(video_path: str) -> Dict[str, Any]:
    """
    Analyze audio from video for potential cheating indicators.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        Dict[str, Any]: Analysis results including:
            - anomaly_count: Number of audio anomalies detected
            - anomaly_timestamps: List of timestamps where anomalies occurred
            - suspicion_score: Score indicating likelihood of cheating (0-100)
            - transcript: Text transcript of the audio
    """
    try:
        # Initialize results
        results = {
            'anomaly_count': 0,
            'anomaly_timestamps': [],
            'suspicion_score': 0,
            'transcript': ''
        }
        
        # Create analyzer instance
        analyzer = AudioAnalyzer()
        
        # Extract audio from video
        audio_path = analyzer.extract_audio(video_path)
        
        try:
            # Load Whisper model for speech recognition
            model = whisper.load_model("base")
            
            # Transcribe audio
            transcription = model.transcribe(audio_path)
            results['transcript'] = transcription['text']
            
            # Initialize speaker diarization
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=os.getenv("HF_TOKEN")  # Use environment variable
            )
            
            # Run speaker diarization
            diarization = pipeline(audio_path)
            
            # Analyze speaker patterns
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })
            
            # Detect anomalies
            anomalies = []
            
            # Check for multiple speakers
            unique_speakers = len(set(seg['speaker'] for seg in speaker_segments))
            if unique_speakers > 1:
                results['anomaly_count'] += 1
                anomalies.append({
                    'type': 'multiple_speakers',
                    'timestamp': speaker_segments[0]['start'],
                    'confidence': 0.8
                })
            
            # Check for suspicious keywords in transcript
            suspicious_keywords = ['answer', 'cheat', 'help', 'copy', 'google']
            for word in suspicious_keywords:
                if word.lower() in results['transcript'].lower():
                    results['anomaly_count'] += 1
                    # Find timestamp of the word
                    word_index = results['transcript'].lower().find(word.lower())
                    timestamp = word_index / len(results['transcript']) * len(speaker_segments)
                    anomalies.append({
                        'type': 'suspicious_keyword',
                        'timestamp': timestamp,
                        'confidence': 0.7
                    })
            
            # Calculate suspicion score
            base_score = min(results['anomaly_count'] * 20, 100)  # 20 points per anomaly, max 100
            results['suspicion_score'] = base_score
            
            # Add timestamps of anomalies
            results['anomaly_timestamps'] = [anomaly['timestamp'] for anomaly in anomalies]
            
        finally:
            # Cleanup
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
        return results
        
    except Exception as e:
        logger.error(f"Error in audio analysis: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the module
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        results = detect_audio_anomalies(video_path)
        print(json.dumps(results, indent=2))

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 