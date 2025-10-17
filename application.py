from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from datetime import datetime
import logging
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for Node.js backend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hugging Face Inference API Configuration
HF_API_URL = "https://api-inference.huggingface.co/models/SamLowe/roberta-base-go_emotions"
HF_API_TOKEN = os.environ.get('HUGGINGFACE_TOKEN', '')

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 
    'optimism', 'pride', 'realization', 'relief', 'remorse', 
    'sadness', 'surprise', 'neutral'
]

def call_huggingface_api(text):
    """Call Hugging Face Inference API for emotion detection"""
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Hugging Face API: {str(e)}")
        raise


def analyze_text(text, top_k=5, threshold=0.3):
    """Analyze text and return emotion scores"""
    # Call Hugging Face API
    api_result = call_huggingface_api(text)
    
    # API returns list of lists, take first result
    if isinstance(api_result, list) and len(api_result) > 0:
        scores_list = api_result[0]
    else:
        scores_list = api_result
    
    # Convert to our format
    emotion_scores = []
    for item in scores_list:
        emotion_scores.append({
            "emotion": item["label"],
            "score": item["score"]
        })
    
    # Sort by score
    emotion_scores = sorted(emotion_scores, key=lambda x: x['score'], reverse=True)
    
    # Filter by threshold and limit
    filtered_emotions = [e for e in emotion_scores if e['score'] >= threshold][:top_k]
    
    return {
        "primary_emotion": emotion_scores[0]['emotion'],
        "top_emotions": filtered_emotions,
        "all_scores": emotion_scores
    }


def generate_tags(text, max_tags=5):
    emotion_result = analyze_text(text, top_k=3)
    emotion_tags = [e['emotion'] for e in emotion_result['top_emotions']]
    
    # Extract content tags
    words = text.lower().split()
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'were', 'been', 'be',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    
    word_freq = {}
    for word in words:
        word = ''.join(c for c in word if c.isalnum())
        if len(word) > 3 and word not in stopwords:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    content_tags = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
    content_tags = [tag[0] for tag in content_tags]
    
    all_tags = emotion_tags + content_tags
    return all_tags[:max_tags]


def analyze_journal_entry(title, content):
    full_text = f"{title}. {content}"
    emotion_analysis = analyze_text(full_text)
    tags = generate_tags(full_text)
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "title": title,
        "primary_emotion": emotion_analysis['primary_emotion'],
        "emotion_confidence": emotion_analysis['top_emotions'][0]['score'],
        "detected_emotions": emotion_analysis['top_emotions'],
        "tags": tags,
        "emotional_state_summary": get_emotional_summary(emotion_analysis['primary_emotion'])
    }
    
    return result


def get_emotional_summary(primary_emotion):
    emotion_mapping = {
        'joy': 'Happy and positive',
        'sadness': 'Feeling down',
        'anger': 'Frustrated or angry',
        'fear': 'Anxious or worried',
        'excitement': 'Excited and energized',
        'love': 'Loving and warm',
        'gratitude': 'Grateful and appreciative',
        'nervousness': 'Nervous and uneasy',
        'optimism': 'Hopeful and optimistic',
        'disappointment': 'Disappointed',
        'confusion': 'Confused or uncertain',
        'caring': 'Caring and compassionate',
        'neutral': 'Calm and neutral',
        'grief': 'Experiencing deep loss and sorrow',
        'admiration': 'Experiencing admiration',
        'amusement': 'Experiencing amusement',
        'annoyance': 'Feeling irritated',
        'approval': 'Feeling supportive',
        'desire': 'Feeling longing or desire',
        'disapproval': 'Feeling critical',
        'disgust': 'Feeling disgusted',
        'embarrassment': 'Feeling embarrassed',
        'pride': 'Feeling proud',
        'realization': 'Having a realization',
        'relief': 'Feeling relieved',
        'remorse': 'Feeling regretful',
        'surprise': 'Experiencing surprise'
    }
    return emotion_mapping.get(primary_emotion, f"Experiencing {primary_emotion}")

#Routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Eunoia ML Analysis Service (HF API)",
        "model": "SamLowe/roberta-base-go_emotions",
        "method": "Hugging Face Inference API",
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/analyze', methods=['POST'])
def analyze_journal():
    try:
        data = request.get_json()
        
        if not data or 'content' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'content' field"
            }), 400
        
        title = data.get('title', 'Untitled Entry')
        content = data.get('content', '')
        
        if not content.strip():
            return jsonify({
                "success": False,
                "error": "Content cannot be empty"
            }), 400
        
        logger.info(f"Analyzing journal entry: {title[:50]}...")
        
        # Analyze the entry
        result = analyze_journal_entry(title, content)
        
        logger.info(f"Analysis complete. Primary emotion: {result['primary_emotion']}")
        
        return jsonify({
            "success": True,
            "data": result
        }), 200
        
    except Exception as e:
        logger.error(f"Error analyzing journal: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    try:
        data = request.get_json()
        
        if not data or 'entries' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'entries' field"
            }), 400
        
        entries = data.get('entries', [])
        
        if not isinstance(entries, list):
            return jsonify({
                "success": False,
                "error": "'entries' must be an array"
            }), 400
        
        results = []
        for entry in entries:
            title = entry.get('title', 'Untitled')
            content = entry.get('content', '')
            
            if content.strip():
                result = analyze_journal_entry(title, content)
                results.append(result)
        
        return jsonify({
            "success": True,
            "data": results,
            "count": len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/analyze-text', methods=['POST'])
def analyze_text_endpoint():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'text' field"
            }), 400
        
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({
                "success": False,
                "error": "Text cannot be empty"
            }), 400
        
        result = analyze_text(text)
        
        return jsonify({
            "success": True,
            "data": result
        }), 200
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Route not found"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Starting Eunoia ML Service on port {port}")
    # logger.info("Using Hugging Face Inference API (no local model)")
    app.run(host='0.0.0.0', port=port, debug=False)
