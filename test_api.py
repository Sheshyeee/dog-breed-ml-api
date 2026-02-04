#!/usr/bin/env python3
"""
Quick test script to verify ML API is working correctly
Usage: python test_api.py <path_to_dog_image.jpg>
"""

import sys
import requests
import json

def test_ml_api(image_path, base_url="http://localhost:8001"):
    """Test the ML API with an image."""
    
    print(f"🧪 Testing ML API at {base_url}")
    print("="*60)
    
    # Test 1: Health Check
    print("\n1️⃣  Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data['status']}")
            print(f"✅ Model Loaded: {data['model_loaded']}")
            print(f"✅ References Count: {data['references_count']}")
            print(f"✅ Unique Breeds: {data['unique_breeds']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to ML API: {e}")
        print("💡 Make sure the API is running: uvicorn main:app --reload")
        return False
    
    # Test 2: Prediction
    print("\n2️⃣  Testing Breed Prediction...")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: {data['success']}")
            print(f"✅ Breed: {data['breed']}")
            print(f"✅ Confidence: {data['confidence']:.2%}")
            print(f"✅ Is Memory Match: {data['is_memory_match']}")
            print(f"\n📊 Top 5 Predictions:")
            for i, pred in enumerate(data['top_5'], 1):
                conf_pct = pred['confidence'] * 100 if isinstance(pred['confidence'], float) and pred['confidence'] <= 1 else pred['confidence']
                print(f"   {i}. {pred['breed']}: {conf_pct:.2f}% ({pred.get('source', 'unknown')})")
            
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(response.text)
            return False
    except FileNotFoundError:
        print(f"❌ Image file not found: {image_path}")
        return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False
    
    # Test 3: Memory Stats
    print("\n3️⃣  Testing Memory Stats...")
    try:
        response = requests.get(f"{base_url}/memory/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Total Examples: {data['total_examples']}")
            print(f"✅ Unique Breeds: {data['unique_breeds']}")
            if data['breeds']:
                print(f"\n📚 Learned Breeds:")
                for breed, count in list(data['breeds'].items())[:5]:
                    print(f"   - {breed}: {count} examples")
        else:
            print(f"⚠️  Memory stats unavailable")
    except Exception as e:
        print(f"⚠️  Memory stats error: {e}")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("\n💡 Tips:")
    print("   - API Docs: http://localhost:8001/docs")
    print("   - Try learning: POST /learn with file + breed name")
    print("   - Monitor logs in the terminal running uvicorn")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <path_to_dog_image.jpg>")
        print("\nExample:")
        print("  python test_api.py dog.jpg")
        print("  python test_api.py ../test_images/golden_retriever.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8001"
    
    success = test_ml_api(image_path, base_url)
    sys.exit(0 if success else 1)
