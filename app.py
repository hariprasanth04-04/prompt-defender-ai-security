from flask import Flask, render_template, request, jsonify
from markupsafe import escape
import re
import datetime
import hashlib
import os
from collections import defaultdict, deque

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# =========================
# OPTIONAL ML LAYER
# =========================
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    ML_AVAILABLE = True
    print("✅ ML layer available")
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML layer not available. Running in rule-based mode.")

ml_model = None
tokenizer = None

MODEL_DIR = os.environ.get('MODEL_DIR', './enhanced_results')

if ML_AVAILABLE:
    try:
        if os.path.exists(MODEL_DIR):
            ml_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            ml_model.eval()
            print("✅ Custom ML model loaded")
        else:
            print("ℹ️ No custom model found. ML inactive.")
            ml_model = None
            tokenizer = None
    except Exception as e:
        print("⚠️ ML load failed:", e)
        ml_model = None
        tokenizer = None

# =========================
# LOGGING
# =========================
MAX_LOGS = 1000
security_logs_data = deque(maxlen=MAX_LOGS)
log_summary = {'total_blocked': 0}

# =========================
# TEXT PREPROCESSING
# =========================
def preprocess_text(text):
    if not text:
        return ""
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text

# =========================
# CRITICAL PATTERNS
# =========================
CRITICAL_PATTERNS = [
    r"\b(write|create|build)\s+(virus|malware|ransomware|trojan)",
    r"\b(hack|crack|breach|exploit)\s+(system|server|account|website)",
    r"\bddos attack\b",
    r"\brm -rf /\b",
]

def is_critical_threat(prompt):
    normalized = preprocess_text(prompt)
    for pattern in CRITICAL_PATTERNS:
        if re.search(pattern, normalized):
            return True
    return False

# =========================
# RULE-BASED SCORING
# =========================
THREAT_KEYWORDS = [
    'virus','malware','ransomware','trojan',
    'hack','crack','breach','exploit','ddos'
]

def rule_based_score(prompt):
    normalized = preprocess_text(prompt)
    words = normalized.split()
    total = max(len(words), 1)
    matches = sum(1 for w in THREAT_KEYWORDS if w in normalized)
    return min(matches / total, 1.0)

# =========================
# ML PREDICTION (SAFE)
# =========================
def ml_classifier_predict(prompt):
    if not ML_AVAILABLE or ml_model is None or tokenizer is None:
        return rule_based_score(prompt)

    try:
        processed = preprocess_text(prompt)
        inputs = tokenizer(
            processed,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = ml_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            return probs[0][1].item()
    except Exception:
        return rule_based_score(prompt)

# =========================
# ANALYSIS PIPELINE
# =========================
def analyze_prompt(user_input):
    if not user_input:
        return {
            "blocked": False,
            "response": "Provide a prompt.",
            "confidence": 0.0
        }

    if is_critical_threat(user_input):
        log_summary['total_blocked'] += 1
        return {
            "blocked": True,
            "response": "Critical malicious pattern detected.",
            "confidence": 1.0
        }

    score = ml_classifier_predict(user_input)

    if score > 0.8:
        log_summary['total_blocked'] += 1
        return {
            "blocked": True,
            "response": "Blocked due to potential malicious content.",
            "confidence": round(score, 3)
        }

    return {
        "blocked": False,
        "response": "Prompt is safe.",
        "confidence": round(score, 3)
    }

# =========================
# ROUTES
# =========================
@app.route('/')
def dashboard():
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
        "ml_available": ML_AVAILABLE,
        "model_loaded": ml_model is not None,
        "total_blocked": log_summary['total_blocked']
    })

# =========================
# PRODUCTION START
# =========================
if __name__ == '__main__':
    print("🚀 Starting PromptDefender (Production Mode)")
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False
    )
