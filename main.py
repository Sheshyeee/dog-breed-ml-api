#!/usr/bin/env python3
"""
Dog Breed ML API - FastAPI Service
Handles breed prediction with YOLO model + Gemini hybrid verification,
learning/memory system, and age simulation support.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
import json
import os
import io
import logging
import hashlib
import base64
import requests
from datetime import datetime
from collections import defaultdict
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dog Breed ML API",
    description="Machine Learning API for dog breed identification (YOLO + Gemini hybrid verification)",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://gloomily-meritorious-giuseppe.ngrok-free.dev",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"  # Remove in production
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

REFERENCES_FILE = "references.json"

# ============================================================================
# HYBRID-PRONE BREED LIST
# Breeds commonly used as one parent in recognized designer crosses.
# When the YOLO model outputs one of these breeds, Gemini is ALWAYS asked
# to verify whether the dog is actually a hybrid of that breed.
# This bypasses the confidence score problem entirely.
# ============================================================================
HYBRID_PRONE_BREEDS = {
    # Poodle crosses (most common)
    "toy poodle", "miniature poodle", "standard poodle",
    # Spaniel crosses
    "cocker spaniel",
    # Retriever crosses
    "golden retriever", "labrador retriever",
    # Small dog crosses
    "maltese dog", "shih-tzu", "yorkshire terrier", "pomeranian",
    "chihuahua", "pug", "papillon",
    # Terrier crosses
    "soft-coated wheaten terrier", "west highland white terrier",
    # Other popular cross parents
    "bichon frise", "cavalier king charles spaniel",
    "australian shepherd", "bernese mountain dog",
    "old english sheepdog", "border collie",
}

# ============================================================================
# GEMINI CONFIGURATION
# ============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_FLASH_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001:generateContent"

# ============================================================================
# GLOBAL MODEL
# ============================================================================
YOLO_MODEL = None
BREED_MAPPING = {}

# ============================================================================
# DETERMINISTIC HELPERS
# ============================================================================

def calculate_image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()

def get_deterministic_variation(hash_value: str, min_val: float, max_val: float) -> float:
    hash_int = int(hash_value[:16], 16)
    normalized = (hash_int % 1000000) / 1000000.0
    return min_val + (normalized * (max_val - min_val))

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
# MODEL LOADING
# ============================================================================

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
            logger.info(f"✓ Loaded {len(breed_mapping)} breeds from breed_mapping.json")
            return breed_mapping
    except Exception as e:
        logger.error(f"Failed to load breed mapping: {str(e)}")
        raise

def load_ml_model():
    """Load the YOLO model (best.pt) on startup."""
    global YOLO_MODEL
    try:
        from ultralytics import YOLO
        load_breed_mapping()
        YOLO_MODEL = YOLO('best.pt')
        logger.info("✓ YOLO model (best.pt) loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {str(e)}")
        raise

# ============================================================================
# YOLO PREDICTION HELPERS
# ============================================================================

def predict_with_yolo(image_bytes: bytes) -> Dict[str, Any]:
    """
    Run YOLO classification on image bytes.
    Returns top-5 predictions with breed names and confidences (0-1 scale).
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # YOLO classify expects a PIL image or path
    results = YOLO_MODEL(image, verbose=False)

    # For classification models, results[0].probs holds the probabilities
    probs = results[0].probs
    top5_indices = probs.top5          # list of int indices
    top5_confs   = probs.top5conf.tolist()  # list of float confidences

    top5 = []
    for idx, conf in zip(top5_indices, top5_confs):
        breed_name = BREED_MAPPING.get(str(idx), f"Unknown_{idx}")
        top5.append({"breed": breed_name, "confidence": float(conf)})

    return {
        "breed":      top5[0]["breed"],
        "confidence": top5[0]["confidence"],
        "top5":       top5,
    }

def get_yolo_embedding(image_bytes: bytes) -> np.ndarray:
    """
    Extract a feature embedding from YOLO for memory/learning system.
    We use the raw probability vector over all classes as the embedding
    (consistent, deterministic, no extra model needed).
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    results = YOLO_MODEL(image, verbose=False)
    probs_tensor = results[0].probs.data  # full probability vector
    return probs_tensor.cpu().numpy().flatten()

# ============================================================================
# GEMINI HYBRID VERIFICATION
# ============================================================================

def verify_hybrid_with_gemini(image_bytes: bytes, yolo_breed: str, yolo_conf: float) -> Dict[str, Any]:
    """
    Ask Gemini Flash to verify whether a YOLO prediction for a hybrid-prone
    breed is actually a purebred or a recognized hybrid (e.g. Cockapoo).

    Returns a dict:
        {
            "verified": True/False,   # True = Gemini agrees with YOLO
            "final_breed": str,       # Gemini's verdict (may differ from YOLO)
            "is_hybrid": True/False,
            "hybrid_name": str|None,
            "gemini_confidence": float,
            "used_gemini": True,
        }
    """
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set — skipping hybrid verification")
        return {"verified": True, "final_breed": yolo_breed, "is_hybrid": False,
                "hybrid_name": None, "gemini_confidence": yolo_conf, "used_gemini": False}

    try:
        # Encode image
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image = Image.open(io.BytesIO(image_bytes))
        mime = "image/jpeg"
        if image.format == "PNG":
            mime = "image/png"
        elif image.format == "WEBP":
            mime = "image/webp"

        prompt = f"""You are a canine breed expert.

My image classification model predicts this dog is a "{yolo_breed}" with {round(yolo_conf*100, 1)}% confidence.

TASK: Analyze the image carefully and answer:
1. Do you agree this is a purebred {yolo_breed}? 
2. Could this be a recognized hybrid or designer breed that LOOKS LIKE a {yolo_breed}? 
   Common examples: Cockapoo (Cocker Spaniel × Poodle), Cavapoo (Cavalier × Poodle), 
   Maltipoo (Maltese × Poodle), Goldendoodle (Golden Retriever × Poodle), 
   Labradoodle (Labrador × Poodle), Schnoodle (Schnauzer × Poodle), etc.

Look for hybrid indicators:
- Wavy or loosely curled coat (Poodle gene)
- Mixed proportions from two different breed types
- Coat that doesn't match the pure breed standard exactly

Respond ONLY with valid JSON (no markdown, no explanation):
{{"agree": true_or_false, "final_breed": "exact breed or hybrid name", "is_hybrid": true_or_false, "hybrid_name": "hybrid name or null", "confidence": 0_to_100}}

Rules:
- If purebred: agree=true, final_breed="{yolo_breed}", is_hybrid=false, hybrid_name=null
- If recognized hybrid: agree=false, final_breed="HybridName", is_hybrid=true, hybrid_name="HybridName"
- confidence = your certainty (65-98)
"""

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime, "data": image_b64}}
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 200,
            }
        }

        resp = requests.post(
            f"{GEMINI_FLASH_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=30
        )

        if resp.status_code != 200:
            logger.warning(f"Gemini verification returned {resp.status_code} — trusting YOLO")
            return {"verified": True, "final_breed": yolo_breed, "is_hybrid": False,
                    "hybrid_name": None, "gemini_confidence": yolo_conf, "used_gemini": False}

        data = resp.json()
        raw_text = ""
        for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
            if "text" in part and not part.get("thought"):
                raw_text = part["text"].strip()
                break

        # Strip markdown fences if present
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw_text)

        final_breed    = parsed.get("final_breed", yolo_breed).strip()
        is_hybrid      = bool(parsed.get("is_hybrid", False))
        hybrid_name    = parsed.get("hybrid_name") or None
        gemini_conf    = float(parsed.get("confidence", yolo_conf * 100)) / 100.0
        agree          = bool(parsed.get("agree", True))

        logger.info(f"✓ Gemini hybrid verification: agree={agree}, final_breed={final_breed}, "
                    f"is_hybrid={is_hybrid}, confidence={gemini_conf:.2f}")

        return {
            "verified":          agree,
            "final_breed":       final_breed if not agree else yolo_breed,
            "is_hybrid":         is_hybrid,
            "hybrid_name":       hybrid_name,
            "gemini_confidence": gemini_conf,
            "used_gemini":       True,
        }

    except Exception as e:
        logger.error(f"Gemini hybrid verification failed: {e} — trusting YOLO")
        return {"verified": True, "final_breed": yolo_breed, "is_hybrid": False,
                "hybrid_name": None, "gemini_confidence": yolo_conf, "used_gemini": False}

# ============================================================================
# MEMORY / LEARNING HELPERS  (unchanged logic, adapted to numpy embeddings)
# ============================================================================

def euclidean_distance(emb1, emb2):
    return float(np.linalg.norm(emb1 - emb2))

def cosine_similarity(emb1, emb2):
    dot_product = np.dot(emb1, emb2)
    norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    if norm_product == 0:
        return 0.0
    return float(dot_product / norm_product)

def calculate_combined_similarity(emb1, emb2):
    euclidean_dist = euclidean_distance(emb1, emb2)
    cosine_sim = cosine_similarity(emb1, emb2)
    euclidean_score = 1.0 / (1.0 + euclidean_dist / 10.0)
    combined_score = (euclidean_score * 0.4) + (cosine_sim * 0.6)
    return euclidean_dist, cosine_sim, combined_score

def analyze_memory_matches(current_emb, references, image_hash: str):
    breed_matches = defaultdict(list)
    exact_correction_breed = None

    for ref in references:
        ref_emb = np.array(ref['embedding'])
        ref_label = ref['label']
        ref_hash = ref.get('image_hash', '')

        if ref_hash and ref_hash == image_hash:
            exact_correction_breed = ref_label
            logger.info(f"🎯 EXACT MATCH FOUND! Corrected to: {ref_label}")

        euclidean_dist, cosine_sim, combined_score = calculate_combined_similarity(current_emb, ref_emb)

        match_info = {
            'euclidean_distance': euclidean_dist,
            'cosine_similarity':  cosine_sim,
            'combined_score':     combined_score,
            'source_image':       ref.get('source_image', 'unknown'),
            'added_at':           ref.get('added_at', 'unknown'),
            'is_exact_image':     (ref_hash == image_hash),
        }
        breed_matches[ref_label].append(match_info)

    breed_stats = {}
    for breed, matches in breed_matches.items():
        if not matches:
            continue
        euclidean_dists  = [m['euclidean_distance'] for m in matches]
        cosine_sims      = [m['cosine_similarity']   for m in matches]
        combined_scores  = [m['combined_score']       for m in matches]

        breed_stats[breed] = {
            'num_examples':         len(matches),
            'best_match':           min(matches, key=lambda x: x['euclidean_distance']),
            'avg_euclidean':        float(np.mean(euclidean_dists)),
            'min_euclidean':        float(np.min(euclidean_dists)),
            'std_euclidean':        float(np.std(euclidean_dists)) if len(matches) > 1 else 0.0,
            'avg_cosine':           float(np.mean(cosine_sims)),
            'max_cosine':           float(np.max(cosine_sims)),
            'avg_combined_score':   float(np.mean(combined_scores)),
            'max_combined_score':   float(np.max(combined_scores)),
            'all_matches':          matches,
            'is_exact_correction':  any(m['is_exact_image'] for m in matches),
        }

    return breed_stats, exact_correction_breed

def calculate_memory_confidence(breed_stats, breed_name, image_hash: str):
    stats = breed_stats.get(breed_name, {})
    if not stats:
        return get_deterministic_variation(image_hash, 0.45, 0.55)

    if stats.get('is_exact_correction', False):
        return 1.00

    num_examples  = stats['num_examples']
    min_euclidean = stats['min_euclidean']
    max_cosine    = stats['max_cosine']
    std_euclidean = stats['std_euclidean']
    max_combined  = stats['max_combined_score']

    if min_euclidean < EXACT_DUPLICATE_THRESHOLD:
        if min_euclidean < 0.1:   base_conf = 1.00
        elif min_euclidean < 0.5: base_conf = 0.995
        elif min_euclidean < 1.0: base_conf = 0.99
        elif min_euclidean < 2.0: base_conf = 0.985
        elif min_euclidean < 3.5: base_conf = 0.97
        else:                     base_conf = 0.96 - ((min_euclidean - 3.5) / 1.5) * 0.01
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

    example_boost = 0.0
    if base_conf < 0.98:
        if num_examples >= 5:   example_boost = 0.08
        elif num_examples >= 3: example_boost = 0.05
        elif num_examples >= 2: example_boost = 0.03

    variance_penalty = 0.0
    if num_examples >= MIN_EXAMPLES_FOR_STATS and min_euclidean >= EXACT_DUPLICATE_THRESHOLD:
        if std_euclidean > 5.0:   variance_penalty = -0.05
        elif std_euclidean > 3.0: variance_penalty = -0.03

    return min(1.00, max(0.55, base_conf + example_boost + variance_penalty))

def make_weighted_decision(model_breed, model_conf, breed_stats, model_top5, image_hash: str, exact_correction_breed):
    if exact_correction_breed:
        logger.info(f"🎯 Using exact correction: {exact_correction_breed} at 100%")
        decision_info = {
            'method': 'exact_image_correction',
            'decision': 'exact_image_previously_corrected',
            'memory_used': True,
            'agreement': (model_breed == exact_correction_breed),
            'corrected_breed': exact_correction_breed,
        }
        return exact_correction_breed, 1.00, decision_info

    decision_info = {
        'method': 'weighted_scoring',
        'model_breed': model_breed,
        'model_confidence': model_conf,
        'memory_breeds_considered': list(breed_stats.keys()),
        'scores': {},
    }

    model_score = model_conf * 100
    decision_info['scores'][model_breed] = {'source': 'model', 'score': model_score, 'confidence': model_conf}

    memory_candidates = []
    for breed, stats in breed_stats.items():
        min_euclidean = stats['min_euclidean']
        max_cosine    = stats['max_cosine']
        max_combined  = stats['max_combined_score']
        num_examples  = stats['num_examples']

        if min_euclidean > WEAK_SIMILARITY_THRESHOLD and max_cosine < COSINE_WEAK_THRESHOLD:
            continue

        memory_conf = calculate_memory_confidence(breed_stats, breed, image_hash)

        if min_euclidean < EXACT_DUPLICATE_THRESHOLD:
            if min_euclidean < 0.1:   memory_score = 100
            elif min_euclidean < 0.5: memory_score = 99.5
            elif min_euclidean < 1.0: memory_score = 99
            elif min_euclidean < 2.0: memory_score = 98.5
            elif min_euclidean < 3.5: memory_score = 98
            else:                     memory_score = 97
        elif max_cosine > COSINE_EXACT_THRESHOLD:
            memory_score = 96
        elif max_combined > 0.80:
            memory_score = 75 + (max_combined - 0.80) * 50
        else:
            memory_score = 60 + (max_combined - 0.70) * 50

        if num_examples >= 3 and memory_score < 99:
            memory_score += 5
        memory_score = min(100, memory_score)

        memory_candidates.append({'breed': breed, 'score': memory_score, 'confidence': memory_conf, 'stats': stats})
        decision_info['scores'][breed] = {
            'source': 'memory', 'score': memory_score, 'confidence': memory_conf,
            'num_examples': num_examples, 'min_euclidean': min_euclidean, 'max_cosine': max_cosine,
        }

    memory_candidates.sort(key=lambda x: x['score'], reverse=True)

    if not memory_candidates:
        decision_info['decision'] = 'no_memory_matches'
        decision_info['memory_used'] = False
        return model_breed, model_conf, decision_info

    best_memory       = memory_candidates[0]
    best_memory_breed = best_memory['breed']
    best_memory_score = best_memory['score']
    best_memory_conf  = best_memory['confidence']

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
    if not os.path.exists(REFERENCES_FILE):
        return []
    try:
        with open(REFERENCES_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_references(references):
    with open(REFERENCES_FILE, 'w') as f:
        json.dump(references, f, indent=2)

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Dog Breed ML API (YOLO + Gemini hybrid verification)...")
    load_ml_model()
    if GEMINI_API_KEY:
        logger.info("✓ Gemini API key configured — hybrid verification ENABLED")
    else:
        logger.warning("⚠️  GEMINI_API_KEY not set — hybrid verification DISABLED (YOLO only)")
    logger.info("✓ API ready")

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthCheckResponse)
async def health_check():
    references = load_references()
    unique_breeds = len(set([ref['label'] for ref in references])) if references else 0
    return {
        "status": "healthy",
        "model_loaded": YOLO_MODEL is not None,
        "references_count": len(references),
        "unique_breeds": unique_breeds,
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_breed(file: UploadFile = File(...)):
    """
    Predict dog breed.
    Pipeline:
      1. YOLO classification (best.pt)
      2. If top breed is hybrid-prone → Gemini Flash verifies
      3. Memory/learning system applied on top
    """
    try:
        if YOLO_MODEL is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        image_bytes = await file.read()
        image_hash  = calculate_image_hash(image_bytes)
        logger.info(f"📸 Image hash: {image_hash[:16]}...")

        # ── STEP 1: YOLO PREDICTION ──────────────────────────────────────────
        yolo_result  = predict_with_yolo(image_bytes)
        yolo_breed   = yolo_result["breed"]
        yolo_conf    = yolo_result["confidence"]
        yolo_top5    = yolo_result["top5"]

        logger.info(f"🐕 YOLO top prediction: {yolo_breed} ({yolo_conf*100:.1f}%)")

        # ── STEP 2: HYBRID VERIFICATION (if applicable) ──────────────────────
        gemini_info = {
            "used_gemini": False,
            "verified": True,
            "final_breed": yolo_breed,
            "is_hybrid": False,
            "hybrid_name": None,
            "gemini_confidence": yolo_conf,
        }

        yolo_breed_lower = yolo_breed.lower()
        is_hybrid_prone  = any(hp in yolo_breed_lower for hp in HYBRID_PRONE_BREEDS)

        if is_hybrid_prone:
            logger.info(f"⚠️  '{yolo_breed}' is hybrid-prone → sending to Gemini for verification")
            gemini_info = verify_hybrid_with_gemini(image_bytes, yolo_breed, yolo_conf)

        # Determine model-level final breed & confidence
        if gemini_info["used_gemini"] and not gemini_info["verified"]:
            # Gemini disagrees — use Gemini's breed
            model_breed_final = gemini_info["final_breed"]
            model_conf_final  = gemini_info["gemini_confidence"]
            prediction_source = "gemini_correction"
            logger.info(f"🔄 Gemini overrides YOLO: {yolo_breed} → {model_breed_final}")
        else:
            model_breed_final = yolo_breed
            model_conf_final  = yolo_conf
            prediction_source = "yolo"

        # ── STEP 3: MEMORY SYSTEM ────────────────────────────────────────────
        embedding   = get_yolo_embedding(image_bytes)  # prob vector as embedding
        references  = load_references()

        memory_info = {
            "memory_available": len(references) > 0,
            "memory_size": len(references),
            "memory_used": False,
            "gemini_verification": {
                "triggered":    gemini_info["used_gemini"],
                "yolo_breed":   yolo_breed,
                "yolo_conf":    round(yolo_conf * 100, 1),
                "agreed":       gemini_info["verified"],
                "final_breed":  gemini_info["final_breed"],
                "is_hybrid":    gemini_info["is_hybrid"],
                "hybrid_name":  gemini_info["hybrid_name"],
            },
        }

        final_breed      = model_breed_final
        final_confidence = model_conf_final
        is_memory_match  = False

        if references:
            breed_stats, exact_correction_breed = analyze_memory_matches(embedding, references, image_hash)

            if breed_stats:
                memory_breed, memory_conf, decision_info = make_weighted_decision(
                    model_breed_final, model_conf_final, breed_stats, yolo_top5,
                    image_hash, exact_correction_breed
                )

                memory_info.update({
                    "unique_breeds_in_memory": len(breed_stats),
                    "memory_used":             decision_info.get("memory_used", False),
                    "decision":                decision_info["decision"],
                    "decision_details":        decision_info,
                })

                if decision_info.get("memory_used", False):
                    final_breed      = memory_breed
                    final_confidence = memory_conf
                    is_memory_match  = True

        # ── BUILD TOP-5 RESPONSE ─────────────────────────────────────────────
        top_5_response = []
        if is_memory_match:
            top_5_response.append({"breed": final_breed, "confidence": final_confidence, "source": "memory"})
            for pred in yolo_top5:
                if pred["breed"] != final_breed:
                    top_5_response.append({"breed": pred["breed"], "confidence": pred["confidence"], "source": "model"})
                if len(top_5_response) >= 5:
                    break
        else:
            # If Gemini corrected, put gemini result first then YOLO alternatives
            if gemini_info["used_gemini"] and not gemini_info["verified"]:
                top_5_response.append({"breed": final_breed, "confidence": final_confidence, "source": "gemini_correction"})
                for pred in yolo_top5:
                    if pred["breed"] != final_breed:
                        top_5_response.append({"breed": pred["breed"], "confidence": pred["confidence"], "source": "model"})
                    if len(top_5_response) >= 5:
                        break
            else:
                top_5_response = [{"breed": p["breed"], "confidence": p["confidence"], "source": "model"} for p in yolo_top5]

        # Pad to 5 if needed
        while len(top_5_response) < 5:
            top_5_response.append({"breed": "Other Breeds", "confidence": 0.0, "source": "model"})
        top_5_response = top_5_response[:5]

        return {
            "success":         True,
            "breed":           final_breed,
            "confidence":      round(final_confidence, 4),
            "top_5":           top_5_response,
            "is_memory_match": is_memory_match,
            "memory_info":     memory_info,
            "learning_stats": {
                "memory_available":         memory_info.get("memory_available", False),
                "memory_size":              memory_info.get("memory_size", 0),
                "memory_used":              memory_info.get("memory_used", False),
                "unique_breeds_in_memory":  memory_info.get("unique_breeds_in_memory", 0),
                "gemini_triggered":         gemini_info["used_gemini"],
                "gemini_corrected":         gemini_info["used_gemini"] and not gemini_info["verified"],
            },
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/learn", response_model=LearnResponse)
async def learn_breed(
    file: UploadFile = File(...),
    breed: str = Form(...)
):
    """Add a new reference image for breed learning (admin correction)."""
    try:
        if YOLO_MODEL is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        image_bytes = await file.read()
        image_hash  = calculate_image_hash(image_bytes)
        logger.info(f"📚 Learning: {breed} | hash: {image_hash[:16]}...")

        # Use YOLO probability vector as embedding for memory
        embedding       = get_yolo_embedding(image_bytes)
        embedding_list  = embedding.tolist()

        references = load_references()

        # Check for duplicates
        current_emb     = embedding
        closest_dist    = float('inf')
        closest_label   = None
        closest_idx     = None

        for idx, ref in enumerate(references):
            ref_emb = np.array(ref['embedding'])
            dist    = np.linalg.norm(current_emb - ref_emb)
            if dist < closest_dist:
                closest_dist  = dist
                closest_label = ref['label']
                closest_idx   = idx

        if closest_dist < EXACT_DUPLICATE_THRESHOLD:
            if closest_label == breed:
                return {"success": True, "status": "skipped",
                        "message": "Duplicate detected — already in memory", "breed": breed}
            else:
                references[closest_idx]['label']      = breed
                references[closest_idx]['updated_at'] = datetime.now().isoformat()
                references[closest_idx]['image_hash'] = image_hash
                save_references(references)
                return {"success": True, "status": "updated",
                        "message": f"Updated label from {closest_label} to {breed}", "breed": breed}
        else:
            references.append({
                "label":        breed,
                "embedding":    embedding_list,
                "source_image": file.filename,
                "added_at":     datetime.now().isoformat(),
                "image_hash":   image_hash,
            })
            save_references(references)
            return {"success": True, "status": "added",
                    "message": f"Added new reference (distance: {closest_dist:.2f})", "breed": breed}

    except Exception as e:
        logger.error(f"Learning error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Learning failed: {str(e)}")


@app.get("/memory/stats")
async def get_memory_stats():
    """Get statistics about learned breeds."""
    references = load_references()
    if not references:
        return {"total_examples": 0, "unique_breeds": 0, "breeds": {}}
    breed_counts = {}
    for ref in references:
        breed = ref['label']
        breed_counts[breed] = breed_counts.get(breed, 0) + 1
    return {"total_examples": len(references), "unique_breeds": len(breed_counts), "breeds": breed_counts}


@app.delete("/memory/clear")
async def clear_memory():
    """Clear all learned references (admin only)."""
    try:
        save_references([])
        return {"success": True, "message": "Memory cleare successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)