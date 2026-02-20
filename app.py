from flask import Flask, render_template, request, jsonify
from markupsafe import escape
import re
import os
from collections import deque

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# =====================================================
# OPTIONAL ML LAYER (SAFE IMPORT)
# =====================================================
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    ML_AVAILABLE = True
    print("✅ ML layer available")
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML layer NOT available — using rule-based detection")

ml_model = None
tokenizer = None

if ML_AVAILABLE:
    try:
        MODEL_DIR = os.environ.get('MODEL_DIR', './enhanced_results')
        if os.path.exists(MODEL_DIR):
            ml_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            ml_model.eval()
            print("✅ Custom ML model loaded")
    except Exception as e:
        print("⚠️ ML loading failed:", e)
        ml_model = None
        tokenizer = None

# =====================================================
# SIMPLE RULE-BASED DETECTION
# =====================================================
THREAT_PATTERNS = [
    r"\b(write|create|build)\s+(virus|malware|ransomware|trojan)",
    r"\b(hack|crack|breach|exploit)\s+(system|server|account|website)",
    r"\bddos attack\b",
    r"\brm -rf /\b"
]

def preprocess(text):
    return text.lower()

def is_malicious(prompt):
    normalized = preprocess(prompt)
    for pattern in THREAT_PATTERNS:
        if re.search(pattern, normalized):
            return True
    return False

def ml_predict(prompt):
    if not ML_AVAILABLE or ml_model is None or tokenizer is None:
        return 0.0

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = ml_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            return probs[0][1].item()
    except:
        return 0.0

def analyze_prompt(prompt):
    if not prompt:
        return {"blocked": False, "confidence": 0.0}

    if is_malicious(prompt):
        return {"blocked": True, "confidence": 1.0}

    score = ml_predict(prompt)

    if score > 0.8:
        return {"blocked": True, "confidence": round(score, 3)}

    return {"blocked": False, "confidence": round(score, 3)}

# =====================================================
# ROUTES
# =====================================================
@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True) or {}
    prompt = escape(data.get('prompt', ''))
    result = analyze_prompt(prompt)
    return jsonify(result)

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "ml_available": ML_AVAILABLE
    })

# =====================================================
# PRODUCTION START
# =====================================================
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False
    )
