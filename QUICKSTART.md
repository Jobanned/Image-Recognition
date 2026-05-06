# Quick Start - Local Development

## Run Locally

1. **Install dependencies** (in project root):

   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server**:

   ```bash
   uvicorn api.index:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Open in browser**:
   - http://localhost:8000
   - Click "Start Camera"
   - Grant camera permissions
   - Make gestures to see inference results

## Deploy to Vercel

1. **Push to GitHub**:

   ```bash
   git add .
   git commit -m "Convert to web version with FastAPI"
   git push origin main
   ```

2. **Deploy**:
   - Go to https://vercel.com/new
   - Select your GitHub repo
   - Click "Deploy"
   - Done! Your app is live at `your-project.vercel.app`

## What Changed

✅ **Backend**: FastAPI replaces simple HTTP server  
✅ **API Endpoint**: `/api/analyze` accepts base64 frames  
✅ **Frontend**: Browser camera (getUserMedia) streams frames to backend  
✅ **Deployment**: Ready for Vercel serverless  
✅ **Model**: MediaPipe runs server-side for consistency

## Gesture Recognition

- **🤔 THINK**: Index finger near mouth
- **☝️ KNOW**: Index up, fingers down
- **😶 IDLE**: No gesture detected
