import os
import json
from typing import Any, Dict, Optional
from pathlib import Path
import yaml

class Config:
    """Configuration management class"""
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config_data: Dict[str, Any] = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        file_ext = os.path.splitext(config_path)[1].lower()
        
        try:
            with open(config_path, 'r') as f:
                if file_ext == '.json':
                    self.config_data = json.load(f)
                elif file_ext in ['.yaml', '.yml']:
                    self.config_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_ext}")
        except Exception as e:
            raise Exception(f"Error loading config file: {str(e)}")
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save configuration to file"""
        if config_path is None:
            config_path = self.config_path
            
        if config_path is None:
            raise ValueError("No config path specified")
            
        file_ext = os.path.splitext(config_path)[1].lower()
        
        try:
            with open(config_path, 'w') as f:
                if file_ext == '.json':
                    json.dump(self.config_data, f, indent=4)
                elif file_ext in ['.yaml', '.yml']:
                    yaml.dump(self.config_data, f, default_flow_style=False)
                else:
                    raise ValueError(f"Unsupported config file format: {file_ext}")
        except Exception as e:
            raise Exception(f"Error saving config file: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config_data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config_data[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update multiple configuration values"""
        self.config_data.update(config_dict)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return self.config_data.copy()

def load_default_config() -> Config:
    """Load default configuration"""
    default_config = {
        'video': {
            'sample_interval': 0.5,
            'scene_change_threshold': 30.0,
            'output_format': 'mp4',
            'fps': 30
        },
        'audio': {
            'sample_rate': 16000,
            'channels': 1,
            'silence_threshold': 0.01
        },
        'detection': {
            'gaze_threshold': 0.5,
            'person_confidence': 0.5,
            'screen_switch_threshold': 0.7
        },
        'paths': {
            'uploads': 'uploads',
            'results': 'results',
            'models': 'models',
            'logs': 'logs'
        },
        'logging': {
            'level': 'INFO',
            'file': True,
            'console': True
        }
    }
    
    config = Config()
    config.update(default_config)
    return config

def create_config_file(config_path: str, config_data: Optional[Dict[str, Any]] = None) -> None:
    """Create a new configuration file"""
    if os.path.exists(config_path):
        raise FileExistsError(f"Config file already exists: {config_path}")
        
    if config_data is None:
        config = load_default_config()
        config_data = config.get_all()
        
    config = Config()
    config.update(config_data)
    config.save_config(config_path)

def get_config_path() -> str:
    """Get the default configuration file path"""
    # First check for config in current directory
    local_config = Path('config.yaml')
    if local_config.exists():
        return str(local_config)
        
    # Then check in user's home directory
    home_config = Path.home() / '.anti-model' / 'config.yaml'
    if home_config.exists():
        return str(home_config)
        
    # Create default config in user's home directory
    home_config.parent.mkdir(parents=True, exist_ok=True)
    create_config_file(str(home_config))
    return str(home_config) 