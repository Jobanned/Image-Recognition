import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from .inference import InferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# CORS configuration - restrict to Vercel domain and localhost
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    os.environ.get("VERCEL_URL", "").replace("vercel.app", "vercel.app"),
]
# Add your custom domain if deployed elsewhere
custom_domain = os.environ.get("CUSTOM_DOMAIN")
if custom_domain:
    allowed_origins.append(f"https://{custom_domain}")

# Filter out empty strings
allowed_origins = [origin for origin in allowed_origins if origin]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],  # Fallback if no origins set
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

class FrameAnalysisRequest(BaseModel):
    """Request body for frame analysis."""
    frame_base64: str


class FrameAnalysisResponse(BaseModel):
    """Response for frame analysis."""
    state: str
    handsDetected: int
    faceDetected: bool
    handLandmarks: list
    handConnections: list


@app.get('/api/health')
async def health_check():
    """Health check endpoint for monitoring."""
    return JSONResponse({'status': 'ok'})


@limiter.limit("30/minute")
@app.post('/api/analyze')
async def analyze_frame(request: Request, payload: FrameAnalysisRequest):
    """
    Analyze a single frame for hand gestures.
    
    Rate limited to 30 requests/minute per IP.
    Expects a base64-encoded image in the request body.
    Returns the detected gesture state and hand landmarks.
    """
    try:
        # Input validation
        if not payload.frame_base64:
            raise HTTPException(status_code=400, detail="frame_base64 is required")
        
        # Validate base64 size (max 2MB for safety)
        max_size = 2_097_152
        if len(payload.frame_base64) > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"Frame too large (max {max_size} bytes)"
            )
        
        # Basic format validation
        if ',' in payload.frame_base64:
            # Has data URI prefix - validate it
            header = payload.frame_base64.split(',')[0]
            if not ('image' in header):
                raise HTTPException(status_code=400, detail="Invalid image format")
        else:
            # Raw base64 - check if it looks like image data
            if payload.frame_base64[:4] not in ['iVBO', '/9j/']:  # PNG or JPEG magic bytes
                logger.warning("Suspicious base64 prefix: %s", payload.frame_base64[:4])
        
        engine = InferenceEngine()
        result = engine.analyze_base64_frame(payload.frame_base64)
        return FrameAnalysisResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error analyzing frame: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail="Frame analysis failed")


# Static files are served by Vercel directly from the root directory.
# This file only handles API requests under /api/*.


