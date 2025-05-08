from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional, List
import json
import logging
from pydantic import BaseModel
import asyncio
from functools import wraps
import time

# Custom response model
class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Request validation model
class VideoRequest(BaseModel):
    video_path: str
    options: Optional[Dict[str, Any]] = None

# API error handler
def handle_api_error(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.error(f"API Error in {func.__name__}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    return wrapper

# API response handler
def create_response(
    success: bool,
    message: str,
    data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
) -> JSONResponse:
    """Create a standardized API response"""
    response = APIResponse(
        success=success,
        message=message,
        data=data,
        error=error
    )
    return JSONResponse(content=response.dict())

# File upload handler
async def handle_file_upload(
    file: UploadFile,
    allowed_extensions: List[str],
    max_size_mb: int = 100
) -> str:
    """Handle file upload with validation"""
    # Check file extension
    file_ext = file.filename.split('.')[-1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check file size
    max_size_bytes = max_size_mb * 1024 * 1024
    file_size = 0
    chunks = []
    
    while chunk := await file.read(8192):
        file_size += len(chunk)
        if file_size > max_size_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {max_size_mb}MB"
            )
        chunks.append(chunk)
    
    # Save file
    file_path = f"uploads/{file.filename}"
    with open(file_path, 'wb') as f:
        for chunk in chunks:
            f.write(chunk)
    
    return file_path

# API rate limiter
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = []
    
    async def check_rate_limit(self) -> bool:
        """Check if request is within rate limit"""
        current_time = time.time()
        
        # Remove old requests
        self.requests = [t for t in self.requests if current_time - t < 60]
        
        if len(self.requests) >= self.requests_per_minute:
            return False
        
        self.requests.append(current_time)
        return True

# API middleware
def setup_api_middleware(app: FastAPI) -> None:
    """Setup API middleware"""
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

# API health check
@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint"""
    return create_response(
        success=True,
        message="API is healthy",
        data={"status": "ok"}
    )

# API documentation
def setup_api_docs(app: FastAPI) -> None:
    """Setup API documentation"""
    app.title = "Anti-Model API"
    app.description = "API for detecting cheating behavior in video recordings"
    app.version = "1.0.0"
    
    # Add tags
    app.openapi_tags = [
        {
            "name": "video",
            "description": "Video processing endpoints"
        },
        {
            "name": "audio",
            "description": "Audio processing endpoints"
        },
        {
            "name": "detection",
            "description": "Detection endpoints"
        }
    ]

# API error responses
def get_error_response(status_code: int, detail: str) -> JSONResponse:
    """Get standardized error response"""
    return create_response(
        success=False,
        message="Error occurred",
        error=detail
    )

# API success responses
def get_success_response(
    message: str,
    data: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """Get standardized success response"""
    return create_response(
        success=True,
        message=message,
        data=data
    ) 