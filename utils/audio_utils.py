import os
import ffmpeg
import numpy as np
from typing import Tuple, List, Optional
import tempfile
import wave
import struct

def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    """Extract audio from video and save as WAV file"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    if output_path is None:
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        output_path = temp_audio.name
        temp_audio.close()
        
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, output_path, acodec='pcm_s16le', ac=1, ar='16k')
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        return output_path
    except ffmpeg.Error as e:
        raise Exception(f"Error extracting audio: {str(e)}")

def get_audio_info(audio_path: str) -> dict:
    """Get audio file metadata"""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    try:
        probe = ffmpeg.probe(audio_path)
        audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
        
        return {
            'duration': float(probe['format']['duration']),
            'sample_rate': int(audio_info['sample_rate']),
            'channels': int(audio_info['channels']),
            'codec': audio_info['codec_name']
        }
    except ffmpeg.Error as e:
        raise Exception(f"Error getting audio info: {str(e)}")

def read_audio_segment(audio_path: str, start_time: float, duration: float) -> np.ndarray:
    """Read a segment of audio data"""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    try:
        out, _ = (
            ffmpeg
            .input(audio_path, ss=start_time, t=duration)
            .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar='16k')
            .run(capture_stdout=True)
        )
        
        return np.frombuffer(out, np.float32)
    except ffmpeg.Error as e:
        raise Exception(f"Error reading audio segment: {str(e)}")

def calculate_energy(audio_data: np.ndarray) -> float:
    """Calculate audio energy"""
    return np.mean(np.abs(audio_data))

def detect_silence(audio_data: np.ndarray, threshold: float = 0.01) -> bool:
    """Detect if audio segment is silence"""
    return calculate_energy(audio_data) < threshold

def split_audio_into_segments(audio_path: str, segment_duration: float = 1.0) -> List[Tuple[float, np.ndarray]]:
    """Split audio into fixed-duration segments"""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    info = get_audio_info(audio_path)
    segments = []
    
    for start_time in np.arange(0, info['duration'], segment_duration):
        audio_data = read_audio_segment(audio_path, start_time, segment_duration)
        segments.append((start_time, audio_data))
        
    return segments

def save_audio_segment(audio_data: np.ndarray, output_path: str, sample_rate: int = 16000) -> str:
    """Save audio segment to WAV file"""
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        # Convert float32 to int16
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Write audio data
        for sample in audio_data:
            wav_file.writeframes(struct.pack('<h', sample))
            
    return output_path 