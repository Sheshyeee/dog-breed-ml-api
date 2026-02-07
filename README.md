---
title: Dog Breed ML API
emoji: 🐕
colorFrom: blue
colorTo: purple
sdk: docker
app_file: main.py
pinned: false
---

# 🐕 Dog Breed ML API - FastAPI Service

Standalone Python FastAPI service for dog breed identification and learning system.

## 🏗️ Architecture

```
Laravel Backend (PHP) ←→ FastAPI ML Service (Python)
     ↑
     │
Mobile App + Web Frontend
```

## 🌟 Features

- **🎯 High-Accuracy Predictions**: ConvNeXt-Large based model for 120 dog breeds
- **🧠 Adaptive Learning**: Memory system that learns from corrections
- **📊 Confidence Scoring**: Detailed prediction confidence with top-5 results
- **🔄 Smart Decision Making**: Combines model predictions with learned patterns

## 📡 API Endpoints

### POST /predict

Upload an image to get breed prediction.

**Request:**

```bash
curl -X POST "https://YOUR_SPACE.hf.space/predict" \
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
    {"breed": "Labrador Retriever", "confidence": 0.0543, "source": "model"}
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
curl -X POST "https://YOUR_SPACE.hf.space/learn" \
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

### GET /health

Check API health status.

### GET /memory/stats

Get learning statistics.

### DELETE /memory/clear

Clear all learned references.

## 🚀 Usage Examples

### JavaScript/Fetch

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const response = await fetch("https://YOUR_SPACE.hf.space/predict", {
  method: "POST",
  body: formData,
});

const data = await response.json();
console.log(`Breed: ${data.breed}, Confidence: ${data.confidence}`);
```

### Python/Requests

```python
import requests

url = "https://YOUR_SPACE.hf.space/predict"
files = {"file": open("dog_image.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(f"Breed: {result['breed']}")
print(f"Confidence: {result['confidence']}")
```

### PHP/Laravel

```php
use Illuminate\Support\Facades\Http;

$response = Http::attach(
    'file', file_get_contents($imagePath), 'dog.jpg'
)->post('https://YOUR_SPACE.hf.space/predict');

$result = $response->json();
echo "Breed: " . $result['breed'];
```

## 🧠 How the Learning System Works

1. **Initial Prediction**: Model makes prediction based on training
2. **Memory Check**: Compares image with learned references
3. **Smart Decision**: Weighs model confidence vs memory similarity
4. **Learning**: Stores corrections to improve future predictions
5. **Adaptive**: Gets better with each correction

## 📊 Model Information

- **Architecture**: ConvNeXt-Large
- **Input Size**: 384x384 pixels
- **Classes**: 120 dog breeds
- **Feature Dimension**: 512-D embeddings
- **Framework**: PyTorch

## 🎯 Supported Breeds

The API recognizes 120 dog breeds including:

- Chihuahua, Golden Retriever, Labrador Retriever
- German Shepherd, Bulldog, Poodle
- Siberian Husky, Beagle, Rottweiler
- And 111 more breeds!

## 🔗 Integration with Your Project

### Laravel Backend

Update your `.env`:

```env
PYTHON_ML_API_URL=https://YOUR_USERNAME-dog-breed-ml-api.hf.space
```

### Frontend JavaScript

```javascript
const API_URL = "https://YOUR_USERNAME-dog-breed-ml-api.hf.space";
```

## 📚 Interactive Documentation

Visit `/docs` for full Swagger/OpenAPI documentation with interactive testing.

## 📝 Performance

- **Response Time**: ~500-1000ms per prediction
- **Memory Usage**: ~2GB (model + runtime)
- **Concurrent Requests**: Supported

## 🔒 Security Notes

- CORS is configured for cross-origin requests
- For production, update `allow_origins` to your specific domains
- Consider adding API key authentication for production use

## 📧 Support

For issues or questions, check the interactive docs at `/docs`

---

**Built with ❤️ using FastAPI, PyTorch, and Hugging Face Spaces**
