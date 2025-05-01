# Combined Video Analysis API

This API combines three different video analysis capabilities:
1. Gaze Detection
2. Face Detection and Recognition
3. Lip Sync Analysis

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the face landmark predictor:
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

## Running the API

Start the API server:
```bash
python api/main.py
```

The API will be available at `http://localhost:5000`

## API Documentation

Once the server is running, you can access the Swagger documentation at:
`http://localhost:5000`

### Endpoints

#### POST /api/analyze
Analyzes a video file for gaze direction, face detection, and lip sync.

**Request:**
- Content-Type: multipart/form-data
- Parameter: video (file)

**Response:**
```json
{
    "gaze_analysis": [list of gaze direction vectors],
    "face_analysis": [list of detected faces with names and encodings],
    "lip_sync_analysis": [list of lip movement scores]
}
```

## Example Usage

Using curl:
```bash
curl -X POST "http://localhost:5000/api/analyze" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@path/to/your/video.mp4"
```

Using Python requests:
```python
import requests

url = "http://localhost:5000/api/analyze"
files = {"video": open("path/to/your/video.mp4", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Notes

- The API processes videos frame by frame
- Results are returned as arrays containing analysis for each frame
- Face recognition requires known faces to be added to the system
- Lip sync analysis provides a confidence score for lip movement 