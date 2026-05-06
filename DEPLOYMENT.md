# BrainRot Tracker - Web Version

A real-time gesture recognition application using MediaPipe hand tracking and OpenCV.

## Architecture

- **Backend**: FastAPI + MediaPipe for hand tracking inference
- **Frontend**: HTML5 Canvas + WebRTC for camera streaming
- **Deployment**: Vercel (serverless)

## Local Development

### Prerequisites

- Python 3.9+
- Camera access
- pip

### Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run locally with uvicorn:

   ```bash
   uvicorn api.index:app --reload
   ```

3. Open http://localhost:8000 in your browser

## Deployment to Vercel

### Prerequisites

- Vercel account (free tier available)
- GitHub repository with your code

### Steps

1. **Connect your GitHub repo to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Select your GitHub repository
   - Vercel auto-detects the Python framework

2. **Configure environment** (if needed):
   - Root directory: `.` (default)
   - Build command: `pip install -r requirements.txt`
   - Output directory: (leave empty for API-only)

3. **Deploy**:
   - Click "Deploy"
   - Vercel builds and deploys automatically

### Important Notes

- **Cold start**: First request may take 5-10 seconds (model loading)
- **Timeouts**: Free tier has 10-second limit; Pro tier has higher limits
- **File size**: MediaPipe model (~200MB) must be included in deployment
- **Browser camera**: Uses WebRTC - requires HTTPS (Vercel provides this)

## API Endpoints

### `POST /api/analyze`

Analyzes a single video frame.

**Request:**

```json
{
  "frame_base64": "data:image/jpeg;base64,..."
}
```

**Response:**

```json
{
  "state": "THINK|KNOW|IDLE",
  "handsDetected": 2,
  "faceDetected": true,
  "handLandmarks": [...],
  "handConnections": [...]
}
```

### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "ok"
}
```

## Gesture Recognition

- **THINK**: Index finger pointing toward mouth
- **KNOW**: Index finger up, middle/ring/pinky down
- **IDLE**: Default state

## Troubleshooting

### Camera access denied

- Enable camera permissions in browser settings
- Use HTTPS (Vercel does this automatically)
- Try a different browser

### Slow inference

- Reduce frame size (frontend can adjust)
- Allow time for model cold-start on first request
- Check network latency

### Model not found error

- Ensure `hand_landmarker.task` is in project root
- Vercel automatically includes all files in deployment

## File Structure

```
├── api/
│   ├── index.py           # FastAPI app (Vercel entry point)
│   ├── inference.py       # MediaPipe inference logic
│   └── __init__.py
├── Monkey/                # Gesture state images
├── index.html             # Frontend
├── styles.css             # Styling
├── hand_landmarker.task   # MediaPipe model
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata
└── vercel.json            # Vercel configuration
```

## Performance Tips

1. **Reduce image quality** for faster inference
2. **Batch requests** if analyzing multiple frames
3. **Cache model** to avoid reloading (singleton pattern implemented)
4. **Use smaller frame size** for faster processing

## Contributing

Follow the AGENTS.md guidelines for code organization and testing.
