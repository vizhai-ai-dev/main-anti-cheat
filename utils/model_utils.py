import os
import torch
import yaml
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import hashlib
import requests
from tqdm import tqdm

class ModelManager:
    """Class for managing ML models"""
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model_info_file = os.path.join(models_dir, "model_info.yaml")
        self.model_info: Dict[str, Any] = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Load model info if exists
        if os.path.exists(self.model_info_file):
            with open(self.model_info_file, 'r') as f:
                self.model_info = yaml.safe_load(f) or {}
    
    def save_model_info(self) -> None:
        """Save model information to file"""
        with open(self.model_info_file, 'w') as f:
            yaml.dump(self.model_info, f, default_flow_style=False)
    
    def get_model_path(self, model_name: str) -> str:
        """Get the path to a model file"""
        return os.path.join(self.models_dir, model_name)
    
    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if a model is downloaded"""
        return os.path.exists(self.get_model_path(model_name))
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a model"""
        return self.model_info.get(model_name)
    
    def download_model(self, model_url: str, model_name: str, model_info: Dict[str, Any]) -> str:
        """Download a model from URL"""
        if self.is_model_downloaded(model_name):
            logging.info(f"Model {model_name} already exists")
            return self.get_model_path(model_name)
        
        model_path = self.get_model_path(model_name)
        
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            
            with open(model_path, 'wb') as f, tqdm(
                desc=f"Downloading {model_name}",
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for data in response.iter_content(block_size):
                    pbar.update(len(data))
                    f.write(data)
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(model_path)
            
            # Save model info
            model_info['hash'] = file_hash
            self.model_info[model_name] = model_info
            self.save_model_info()
            
            return model_path
            
        except Exception as e:
            if os.path.exists(model_path):
                os.remove(model_path)
            raise Exception(f"Error downloading model: {str(e)}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
                
        return sha256_hash.hexdigest()
    
    def verify_model(self, model_name: str) -> bool:
        """Verify model file integrity"""
        if not self.is_model_downloaded(model_name):
            return False
            
        model_info = self.get_model_info(model_name)
        if not model_info or 'hash' not in model_info:
            return False
            
        current_hash = self._calculate_file_hash(self.get_model_path(model_name))
        return current_hash == model_info['hash']
    
    def list_models(self) -> List[str]:
        """List all downloaded models"""
        return list(self.model_info.keys())
    
    def remove_model(self, model_name: str) -> None:
        """Remove a model"""
        if not self.is_model_downloaded(model_name):
            raise FileNotFoundError(f"Model not found: {model_name}")
            
        model_path = self.get_model_path(model_name)
        os.remove(model_path)
        
        if model_name in self.model_info:
            del self.model_info[model_name]
            self.save_model_info()

def load_model(model_path: str, device: Optional[str] = None) -> torch.nn.Module:
    """Load a PyTorch model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    try:
        model = torch.load(model_path, map_location=device)
        model.eval()
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def save_model(model: torch.nn.Module, model_path: str) -> None:
    """Save a PyTorch model"""
    try:
        torch.save(model, model_path)
    except Exception as e:
        raise Exception(f"Error saving model: {str(e)}")

def get_default_models() -> Dict[str, Dict[str, Any]]:
    """Get default model configurations"""
    return {
        'yolov8n-face.pt': {
            'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-face.pt',
            'type': 'face_detection',
            'framework': 'pytorch',
            'version': '1.0.0'
        },
        'gaze_model.pth': {
            'url': 'https://example.com/models/gaze_model.pth',
            'type': 'gaze_tracking',
            'framework': 'pytorch',
            'version': '1.0.0'
        }
    } 