import os
import json
import shutil
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

def ensure_directory(directory: str) -> None:
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_json(data: Any, filepath: str, indent: int = 4) -> None:
    """Save data to JSON file"""
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)

def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    with open(filepath, 'r') as f:
        return json.load(f)

def get_file_info(filepath: str) -> Dict[str, Any]:
    """Get file metadata"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    stat = os.stat(filepath)
    return {
        'size': stat.st_size,
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'extension': os.path.splitext(filepath)[1]
    }

def cleanup_old_files(directory: str, max_age_days: int = 7) -> None:
    """Remove files older than max_age_days"""
    if not os.path.exists(directory):
        return
        
    current_time = datetime.now()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_time = datetime.fromtimestamp(os.path.getctime(filepath))
            age_days = (current_time - file_time).days
            if age_days > max_age_days:
                try:
                    os.remove(filepath)
                    logging.info(f"Removed old file: {filepath}")
                except Exception as e:
                    logging.error(f"Error removing file {filepath}: {str(e)}")

def move_file(source: str, destination: str, overwrite: bool = False) -> str:
    """Move file to destination"""
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source file not found: {source}")
        
    ensure_directory(os.path.dirname(destination))
    
    if os.path.exists(destination) and not overwrite:
        raise FileExistsError(f"Destination file already exists: {destination}")
        
    shutil.move(source, destination)
    return destination

def copy_file(source: str, destination: str, overwrite: bool = False) -> str:
    """Copy file to destination"""
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source file not found: {source}")
        
    ensure_directory(os.path.dirname(destination))
    
    if os.path.exists(destination) and not overwrite:
        raise FileExistsError(f"Destination file already exists: {destination}")
        
    shutil.copy2(source, destination)
    return destination

def list_files(directory: str, extension: Optional[str] = None) -> List[str]:
    """List files in directory with optional extension filter"""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    files = []
    for filename in os.listdir(directory):
        if extension is None or filename.endswith(extension):
            files.append(os.path.join(directory, filename))
            
    return sorted(files)

def get_unique_filename(directory: str, base_name: str, extension: str) -> str:
    """Generate unique filename in directory"""
    ensure_directory(directory)
    counter = 1
    while True:
        filename = f"{base_name}_{counter}{extension}"
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            return filepath
        counter += 1 