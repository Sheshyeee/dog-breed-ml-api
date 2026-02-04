# Dog Breed ML API - FastAPI Service

Standalone Python FastAPI service for dog breed identification and learning system.

## 🏗️ Architecture

```
Laravel Backend (PHP) ←→ FastAPI ML Service (Python)
     ↑
     │
Mobile App + Web Frontend
```

## 📁 Required Files

Place these files from your Laravel `ml/` directory into this folder:

1. `best_model.pth` - Your trained PyTorch model
2. `breed_mapping.json` - Breed index to name mapping
3. `references.json` - Learned breed references (will be created if doesn't exist)

## 🚀 Local Development

### 1. Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Server

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8001

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 4
```

### 3. Test the API

Visit: `http://localhost:8001/docs`

This will show the interactive Swagger UI documentation.

## 📡 API Endpoints

### POST /predict
Upload an image to get breed prediction.

**Request:**
```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@dog_image.jpg"
```

**Response:**
```json
{
  "success": true,
  "breed": "Golden Retriever",
  "confidence": 0.9234,
  "top_5": [
    {"breed": "Golden Retriever", "confidence": 0.9234, "source": "model"},
    {"breed": "Labrador Retriever", "confidence": 0.0543, "source": "model"},
    ...
  ],
  "is_memory_match": false,
  "memory_info": {...},
  "learning_stats": {...}
}
```

### POST /learn
Add a corrected breed to the learning system.

**Request:**
```bash
curl -X POST "http://localhost:8001/learn" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@dog_image.jpg" \
  -F "breed=Aspin"
```

**Response:**
```json
{
  "success": true,
  "status": "added",
  "message": "Added new reference (distance: 15.34)",
  "breed": "Aspin"
}
```

### GET /memory/stats
Get learning statistics.

### DELETE /memory/clear
Clear all learned references.

## 🐳 Docker Deployment

```bash
# Build image
docker build -t dog-breed-ml-api .

# Run container
docker run -p 8001:8001 \
  -v $(pwd)/references.json:/app/references.json \
  dog-breed-ml-api
```

## ☁️ Deploy to Render

1. Push code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click "New +" → "Web Service"
4. Connect your repository
5. Configure:
   - **Name:** `dog-breed-ml-api`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free or Starter
6. Add environment variables:
   - `PYTHON_VERSION`: `3.10.0`
7. Deploy!

**Important:** Upload `best_model.pth` and `breed_mapping.json` as persistent files in Render.

## ☁️ Deploy to Railway

1. Push code to GitHub
2. Go to [Railway](https://railway.app)
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will auto-detect Python and deploy
6. Get your deployment URL (e.g., `https://your-app.railway.app`)

## 🔗 Connecting to Laravel

Once deployed, update your Laravel `.env`:

```env
# Your deployed FastAPI service URL
PYTHON_ML_API_URL=https://dog-breed-ml-api.onrender.com
# or
PYTHON_ML_API_URL=https://your-app.railway.app
```

## 📊 Performance

- **Cold start:** ~2-5 seconds
- **Prediction:** ~300-800ms per image
- **Memory usage:** ~1.5GB (model + runtime)

## 🔒 Security Notes

- In production, replace `allow_origins=["*"]` with your actual domains
- Add API key authentication if needed
- Use HTTPS in production

## 📝 License

Same license as your main project.
