#!/usr/bin/env python3
"""
Dog Breed ML API - FastAPI Service
Handles breed prediction, learning, and age simulation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json
import os
import io
import logging
from datetime import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dog Breed ML API",
    description="Machine Learning API for dog breed identification and age simulation",
    version="1.0.0"
)

# CORS Configuration - Allow your Laravel backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://gloomily-meritorious-giuseppe.ngrok-free.dev",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"  # Remove this in production, specify exact domains
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION
# ============================================================================

EXACT_DUPLICATE_THRESHOLD = 5.0
VERY_SIMILAR_THRESHOLD = 12.0
SIMILAR_THRESHOLD = 20.0
WEAK_SIMILARITY_THRESHOLD = 30.0

COSINE_EXACT_THRESHOLD = 0.98
COSINE_VERY_SIMILAR_THRESHOLD = 0.92
COSINE_SIMILAR_THRESHOLD = 0.85
COSINE_WEAK_THRESHOLD = 0.75

MODEL_VERY_HIGH_CONF = 0.90
MODEL_HIGH_CONF = 0.85
MODEL_MEDIUM_CONF = 0.70

MIN_EXAMPLES_FOR_STATS = 3

# Reference file path (persistent storage)
REFERENCES_FILE = "references.json"

# ============================================================================
# MODELS
# ============================================================================

class EliteDogClassifier(nn.Module):
    """Enhanced ConvNeXt-Large based dog breed classifier."""
    def __init__(self, num_classes=120):
        super().__init__()
        self.backbone = models.convnext_large(weights=None)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Sequential(
            self.backbone.classifier[0], 
            self.backbone.classifier[1],
            nn.Dropout(0.3), 
            nn.Linear(in_features, 512),
            nn.GELU(), 
            nn.Dropout(0.2), 
            nn.Linear(512, num_classes)
        )

    def forward_features(self, x):
        """Extract feature embeddings (512-dim) before final classification."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = self.backbone.classifier[0](x)
        x = self.backbone.classifier[1](x)
        x = self.backbone.classifier[2](x)
        x = self.backbone.classifier[3](x)
        x = self.backbone.classifier[4](x)
        return x

    def forward(self, x):
        """Full forward pass including classification."""
        x = self.forward_features(x)
        x = self.backbone.classifier[5](x)
        x = self.backbone.classifier[6](x)
        return x

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PredictionResponse(BaseModel):
    success: bool
    breed: str
    confidence: float
    top_5: List[Dict[str, Any]]
    is_memory_match: bool
    memory_info: Dict[str, Any]
    learning_stats: Dict[str, Any]

class LearnResponse(BaseModel):
    success: bool
    status: str
    message: str
    breed: str

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    references_count: int
    unique_breeds: int

# ============================================================================
# GLOBAL MODEL LOADING
# ============================================================================

MODEL = None
DEVICE = None
BREED_MAPPING = {}

def load_breed_mapping():
    """Load breed index to name mapping from JSON file."""
    global BREED_MAPPING
    try:
        with open('breed_mapping.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            breeds = data['breeds']
            breed_mapping = {}
            for idx, breed_id in enumerate(breeds):
                breed_name = breed_id.split('-', 1)[1].replace('_', ' ')
                breed_mapping[str(idx)] = breed_name
            BREED_MAPPING = breed_mapping
            logger.info(f"✓ Loaded {len(breed_mapping)} breeds")
            return breed_mapping
    except Exception as e:
        logger.error(f"Failed to load breed mapping: {str(e)}")
        raise

def load_ml_model():
    """Load the trained model on startup."""
    global MODEL, DEVICE
    try:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {DEVICE}")
        
        # Load breed mapping
        load_breed_mapping()
        num_classes = len(BREED_MAPPING)
        
        # Load model
        MODEL = EliteDogClassifier(num_classes=num_classes)
        state_dict = torch.load('best_model.pth', map_location=DEVICE, weights_only=False)
        MODEL.load_state_dict(state_dict)
        MODEL.eval()
        MODEL.to(DEVICE)
        
        logger.info("✓ Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess image for model input."""
    try:
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return transform(image).unsqueeze(0)
    except Exception as e:
        logger.error(f"Failed to preprocess image: {str(e)}")
        raise

def euclidean_distance(emb1, emb2):
    """Calculate Euclidean distance between two embeddings."""
    return float(np.linalg.norm(emb1 - emb2))

def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between two embeddings (0-1 scale)."""
    dot_product = np.dot(emb1, emb2)
    norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    if norm_product == 0:
        return 0.0
    return float(dot_product / norm_product)

def calculate_combined_similarity(emb1, emb2):
    """Calculate both similarity metrics and return a combined score."""
    euclidean_dist = euclidean_distance(emb1, emb2)
    cosine_sim = cosine_similarity(emb1, emb2)
    
    euclidean_score = 1.0 / (1.0 + euclidean_dist / 10.0)
    combined_score = (euclidean_score * 0.4) + (cosine_sim * 0.6)
    
    return euclidean_dist, cosine_sim, combined_score

def analyze_memory_matches(current_emb, references):
    """Analyze all memory matches and group by breed with statistics."""
    breed_matches = defaultdict(list)
    
    for ref in references:
        ref_emb = np.array(ref['embedding'])
        ref_label = ref['label']
        
        euclidean_dist, cosine_sim, combined_score = calculate_combined_similarity(
            current_emb, ref_emb
        )
        
        match_info = {
            'euclidean_distance': euclidean_dist,
            'cosine_similarity': cosine_sim,
            'combined_score': combined_score,
            'source_image': ref.get('source_image', 'unknown'),
            'added_at': ref.get('added_at', 'unknown')
        }
        
        breed_matches[ref_label].append(match_info)
    
    breed_stats = {}
    for breed, matches in breed_matches.items():
        if len(matches) == 0:
            continue
            
        euclidean_dists = [m['euclidean_distance'] for m in matches]
        cosine_sims = [m['cosine_similarity'] for m in matches]
        combined_scores = [m['combined_score'] for m in matches]
        
        breed_stats[breed] = {
            'num_examples': len(matches),
            'best_match': min(matches, key=lambda x: x['euclidean_distance']),
            'avg_euclidean': float(np.mean(euclidean_dists)),
            'min_euclidean': float(np.min(euclidean_dists)),
            'std_euclidean': float(np.std(euclidean_dists)) if len(matches) > 1 else 0.0,
            'avg_cosine': float(np.mean(cosine_sims)),
            'max_cosine': float(np.max(cosine_sims)),
            'avg_combined_score': float(np.mean(combined_scores)),
            'max_combined_score': float(np.max(combined_scores)),
            'all_matches': matches
        }
    
    return breed_stats

def calculate_memory_confidence(breed_stats, breed_name):
    """Calculate confidence score for exact matches properly."""
    stats = breed_stats.get(breed_name, {})
    if not stats:
        return 0.5
    
    num_examples = stats['num_examples']
    min_euclidean = stats['min_euclidean']
    max_cosine = stats['max_cosine']
    std_euclidean = stats['std_euclidean']
    max_combined = stats['max_combined_score']
    
    # Exact duplicate detection with proper confidence scoring
    if min_euclidean < EXACT_DUPLICATE_THRESHOLD:
        if min_euclidean < 0.1:
            base_conf = 1.00
        elif min_euclidean < 0.5:
            base_conf = 0.995
        elif min_euclidean < 1.0:
            base_conf = 0.99
        elif min_euclidean < 2.0:
            base_conf = 0.985
        elif min_euclidean < 3.5:
            base_conf = 0.97
        else:
            base_conf = 0.96 - ((min_euclidean - 3.5) / 1.5) * 0.01
    elif max_cosine > COSINE_EXACT_THRESHOLD:
        base_conf = 0.94
    elif max_combined > 0.90:
        base_conf = 0.88
    elif min_euclidean < VERY_SIMILAR_THRESHOLD and max_cosine > COSINE_VERY_SIMILAR_THRESHOLD:
        base_conf = 0.82
    elif max_combined > 0.80:
        base_conf = 0.75
    else:
        base_conf = 0.65
    
    if base_conf < 0.98:
        if num_examples >= 5:
            example_boost = 0.08
        elif num_examples >= 3:
            example_boost = 0.05
        elif num_examples >= 2:
            example_boost = 0.03
        else:
            example_boost = 0.0
    else:
        example_boost = 0.0
    
    if num_examples >= MIN_EXAMPLES_FOR_STATS and min_euclidean >= EXACT_DUPLICATE_THRESHOLD:
        if std_euclidean > 5.0:
            variance_penalty = -0.05
        elif std_euclidean > 3.0:
            variance_penalty = -0.03
        else:
            variance_penalty = 0.0
    else:
        variance_penalty = 0.0
    
    final_conf = base_conf + example_boost + variance_penalty
    return min(1.00, max(0.55, final_conf))

def make_weighted_decision(model_breed, model_conf, breed_stats, model_top5):
    """Make final prediction with proper handling of exact matches."""
    decision_info = {
        'method': 'weighted_scoring',
        'model_breed': model_breed,
        'model_confidence': model_conf,
        'memory_breeds_considered': list(breed_stats.keys()),
        'scores': {}
    }
    
    model_score = model_conf * 100
    decision_info['scores'][model_breed] = {
        'source': 'model',
        'score': model_score,
        'confidence': model_conf
    }
    
    memory_candidates = []
    for breed, stats in breed_stats.items():
        min_euclidean = stats['min_euclidean']
        max_cosine = stats['max_cosine']
        max_combined = stats['max_combined_score']
        num_examples = stats['num_examples']
        
        if min_euclidean > WEAK_SIMILARITY_THRESHOLD and max_cosine < COSINE_WEAK_THRESHOLD:
            continue
        
        memory_conf = calculate_memory_confidence(breed_stats, breed)
        
        if min_euclidean < EXACT_DUPLICATE_THRESHOLD:
            if min_euclidean < 0.1:
                memory_score = 100
            elif min_euclidean < 0.5:
                memory_score = 99.5
            elif min_euclidean < 1.0:
                memory_score = 99
            elif min_euclidean < 2.0:
                memory_score = 98.5
            elif min_euclidean < 3.5:
                memory_score = 98
            else:
                memory_score = 97
        elif max_cosine > COSINE_EXACT_THRESHOLD:
            memory_score = 96
        elif max_combined > 0.80:
            memory_score = 75 + (max_combined - 0.80) * 50
        else:
            memory_score = 60 + (max_combined - 0.70) * 50
        
        if num_examples >= 3 and memory_score < 99:
            memory_score += 5
        
        memory_score = min(100, memory_score)
        
        memory_candidates.append({
            'breed': breed,
            'score': memory_score,
            'confidence': memory_conf,
            'stats': stats
        })
        
        decision_info['scores'][breed] = {
            'source': 'memory',
            'score': memory_score,
            'confidence': memory_conf,
            'num_examples': num_examples,
            'min_euclidean': min_euclidean,
            'max_cosine': max_cosine
        }
    
    memory_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    if not memory_candidates:
        decision_info['decision'] = 'no_memory_matches'
        decision_info['memory_used'] = False
        return model_breed, model_conf, decision_info
    
    best_memory = memory_candidates[0]
    best_memory_breed = best_memory['breed']
    best_memory_score = best_memory['score']
    best_memory_conf = best_memory['confidence']
    
    if best_memory['stats']['min_euclidean'] < EXACT_DUPLICATE_THRESHOLD:
        decision_info['decision'] = 'memory_exact_duplicate'
        decision_info['memory_used'] = True
        decision_info['agreement'] = (model_breed == best_memory_breed)
        return best_memory_breed, best_memory_conf, decision_info
    
    if model_breed == best_memory_breed:
        agreement_boost = 0.08 if best_memory_score > 85 else 0.05
        final_conf = min(0.98, max(model_conf, best_memory_conf) + agreement_boost)
        decision_info['decision'] = 'agreement_confidence_boost'
        decision_info['memory_used'] = True
        decision_info['agreement'] = True
        return model_breed, final_conf, decision_info
    
    if model_conf > MODEL_VERY_HIGH_CONF and best_memory_score < 92:
        decision_info['decision'] = 'model_very_high_confidence_override'
        decision_info['memory_used'] = False
        decision_info['agreement'] = False
        return model_breed, model_conf, decision_info
    
    margin_required = 15
    
    if best_memory_score > model_score + margin_required:
        decision_info['decision'] = 'memory_score_advantage'
        decision_info['memory_used'] = True
        decision_info['agreement'] = False
        decision_info['score_margin'] = best_memory_score - model_score
        return best_memory_breed, best_memory_conf, decision_info
    
    if model_score > best_memory_score + margin_required:
        decision_info['decision'] = 'model_score_advantage'
        decision_info['memory_used'] = False
        decision_info['agreement'] = False
        decision_info['score_margin'] = model_score - best_memory_score
        return model_breed, model_conf, decision_info
    
    if best_memory_conf > model_conf:
        decision_info['decision'] = 'memory_higher_confidence'
        decision_info['memory_used'] = True
        decision_info['agreement'] = False
        return best_memory_breed, best_memory_conf, decision_info
    else:
        decision_info['decision'] = 'model_higher_confidence'
        decision_info['memory_used'] = False
        decision_info['agreement'] = False
        return model_breed, model_conf, decision_info

def load_references():
    """Load reference embeddings from JSON file."""
    if not os.path.exists(REFERENCES_FILE):
        return []
    try:
        with open(REFERENCES_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_references(references):
    """Save reference embeddings to JSON file."""
    with open(REFERENCES_FILE, 'w') as f:
        json.dump(references, f, indent=2)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("🚀 Starting Dog Breed ML API...")
    load_ml_model()
    logger.info("✓ API ready to serve requests")

@app.get("/", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    references = load_references()
    unique_breeds = len(set([ref['label'] for ref in references])) if references else 0
    
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "references_count": len(references),
        "unique_breeds": unique_breeds
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_breed(file: UploadFile = File(...)):
    """
    Predict dog breed from uploaded image.
    
    - **file**: Image file (JPEG, PNG, etc.)
    
    Returns breed prediction with confidence and top 5 predictions.
    """
    try:
        if MODEL is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Read and preprocess image
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes).to(DEVICE)
        
        # Extract embedding
        with torch.no_grad():
            embedding = MODEL.forward_features(image_tensor)
        
        # Get model predictions
        with torch.no_grad():
            x = MODEL.backbone.classifier[5](embedding)
            outputs = MODEL.backbone.classifier[6](x)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, 5)
        
        model_breed_idx = str(top_indices[0][0].item())
        model_breed_name = BREED_MAPPING.get(model_breed_idx, 'Unknown')
        model_confidence = float(top_probs[0][0].item())
        
        # Build model top 5
        model_top5 = []
        for i in range(len(top_probs[0])):
            breed_idx = str(top_indices[0][i].item())
            breed_name = BREED_MAPPING.get(breed_idx, 'Unknown')
            model_top5.append({
                'breed': breed_name,
                'confidence': float(top_probs[0][i].item())
            })
        
        # Check memory
        references = load_references()
        current_emb = embedding.cpu().detach().numpy().flatten()
        
        memory_info = {
            'memory_available': len(references) > 0,
            'memory_size': len(references),
            'memory_used': False
        }
        
        final_breed = model_breed_name
        final_confidence = model_confidence
        is_memory_match = False
        
        if references:
            breed_stats = analyze_memory_matches(current_emb, references)
            
            if breed_stats:
                memory_breed, memory_conf, decision_info = make_weighted_decision(
                    model_breed_name, model_confidence, breed_stats, model_top5
                )
                
                memory_info.update({
                    'unique_breeds_in_memory': len(breed_stats),
                    'memory_used': decision_info.get('memory_used', False),
                    'decision': decision_info['decision'],
                    'decision_details': decision_info
                })
                
                if decision_info.get('memory_used', False):
                    final_breed = memory_breed
                    final_confidence = memory_conf
                    is_memory_match = True
        
        # Build top 5 response
        top_5_response = []
        if is_memory_match:
            top_5_response.append({
                'breed': final_breed,
                'confidence': final_confidence,
                'source': 'memory'
            })
            for pred in model_top5:
                if pred['breed'] != final_breed:
                    top_5_response.append({
                        'breed': pred['breed'],
                        'confidence': pred['confidence'],
                        'source': 'model'
                    })
                    if len(top_5_response) >= 5:
                        break
        else:
            top_5_response = [{
                'breed': pred['breed'],
                'confidence': pred['confidence'],
                'source': 'model'
            } for pred in model_top5]
        
        return {
            "success": True,
            "breed": final_breed,
            "confidence": round(final_confidence, 4),
            "top_5": top_5_response,
            "is_memory_match": is_memory_match,
            "memory_info": memory_info,
            "learning_stats": {
                "memory_available": memory_info.get('memory_available', False),
                "memory_size": memory_info.get('memory_size', 0),
                "memory_used": memory_info.get('memory_used', False),
                "unique_breeds_in_memory": memory_info.get('unique_breeds_in_memory', 0),
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/learn", response_model=LearnResponse)
async def learn_breed(
    file: UploadFile = File(...),
    breed: str = Form(...)
):
    """
    Add a new reference image for breed learning.
    
    - **file**: Image file
    - **breed**: Correct breed name
    
    Returns learning status (added, updated, or skipped).
    """
    try:
        if MODEL is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Read and preprocess image
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes).to(DEVICE)
        
        # Extract embedding
        with torch.no_grad():
            embedding = MODEL.forward_features(image_tensor)
        
        embedding_array = embedding.cpu().numpy()
        embedding_list = embedding_array.flatten().tolist()
        
        # Load existing references
        references = load_references()
        
        # Check for duplicates
        current_emb = embedding_array.flatten()
        closest_distance = float('inf')
        closest_label = None
        closest_idx = None
        
        for idx, ref in enumerate(references):
            ref_emb = np.array(ref['embedding'])
            dist = np.linalg.norm(current_emb - ref_emb)
            if dist < closest_distance:
                closest_distance = dist
                closest_label = ref['label']
                closest_idx = idx
        
        # Determine action
        if closest_distance < EXACT_DUPLICATE_THRESHOLD:
            if closest_label == breed:
                return {
                    "success": True,
                    "status": "skipped",
                    "message": "Duplicate detected - already in memory",
                    "breed": breed
                }
            else:
                # Update existing reference
                references[closest_idx]['label'] = breed
                references[closest_idx]['updated_at'] = datetime.now().isoformat()
                save_references(references)
                return {
                    "success": True,
                    "status": "updated",
                    "message": f"Updated label from {closest_label} to {breed}",
                    "breed": breed
                }
        else:
            # Add new reference
            references.append({
                "label": breed,
                "embedding": embedding_list,
                "source_image": file.filename,
                "added_at": datetime.now().isoformat()
            })
            save_references(references)
            
            return {
                "success": True,
                "status": "added",
                "message": f"Added new reference (distance: {closest_distance:.2f})",
                "breed": breed
            }
        
    except Exception as e:
        logger.error(f"Learning error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Learning failed: {str(e)}")

@app.get("/memory/stats")
async def get_memory_stats():
    """Get statistics about learned breeds."""
    references = load_references()
    
    if not references:
        return {
            "total_examples": 0,
            "unique_breeds": 0,
            "breeds": {}
        }
    
    breed_counts = {}
    for ref in references:
        breed = ref['label']
        breed_counts[breed] = breed_counts.get(breed, 0) + 1
    
    return {
        "total_examples": len(references),
        "unique_breeds": len(breed_counts),
        "breeds": breed_counts
    }

@app.delete("/memory/clear")
async def clear_memory():
    """Clear all learned references (admin only)."""
    try:
        save_references([])
        return {
            "success": True,
            "message": "Memory cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
