from flask import Flask, render_template, request, jsonify
from markupsafe import escape
import re
import datetime
import random
import joblib
import numpy as np
from collections import defaultdict, deque
import string
import hashlib
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# ---------------- LOGGING ----------------
MAX_LOGS = 1000
security_logs_data = deque(maxlen=MAX_LOGS)
log_summary = {'total_blocked': 0, 'top_reasons': [], 'recent_attempts': []}

# ---------------- LOAD ENHANCED ML MODEL ----------------
MODEL_DIR = os.environ.get('MODEL_DIR', './enhanced_results')

ml_model = None
tokenizer = None

try:
    if os.path.exists(MODEL_DIR):
        ml_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        ml_model.eval()
        print(f"✅ Loaded custom model from {MODEL_DIR}")
    else:
        print("ℹ️ No custom model found. Running in rule-based mode.")
except Exception as e:
    print(f"Warning: could not load model: {e}")
    ml_model = None
    tokenizer = None

# ---------------- TEXT PREPROCESSING ----------------
def preprocess_text(text):
    if not text:
        return ""
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower()
    text = normalize_obfuscation(text)
    text = ' '.join(text.split())
    return text

def normalize_obfuscation(text):
    leet_map = {
        '@': 'a', '$': 's', '0': 'o', '1': 'i', '3': 'e',
        '4': 'a', '7': 't', '!': 'i', '5': 's', '8': 'b',
        '6': 'g', '9': 'g', '+': 't', '|': 'l', '()': 'o'
    }
    for leet, normal in leet_map.items():
        text = text.replace(leet, normal)

    separators = ['-', '_', '.', '*', ' ', '\t', '\n']
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 3:
            text = ''.join(parts)

    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text

# ---------------- CRITICAL THREAT PATTERNS ----------------
CRITICAL_ATTACK_PATTERNS = [
    r"\b(?:build|create|make|develop|generate|code|write|program)\s+(?:a\s+|an\s+)?(?:virus|malware|ransomware|trojan|worm|keylogger|spyware|rootkit|botnet)",
    r"\b(?:hack|crack|exploit|penetrate|breach|compromise)\s+(?:into\s+)?(?:system|server|website|database|network|account)",
    r"\b(?:gain|get)\s+(?:unauthorized\s+)?(?:access|admin|root)",
    r"\b(?:destroy|delete|wipe|erase)\s+(?:all\s+)?(?:files|data|system)",
    r"\b(?:ddos|dos)\s+attack",
]

def is_critical_threat(prompt):
    if not prompt:
        return None
    normalized = preprocess_text(prompt)
    for pattern in CRITICAL_ATTACK_PATTERNS:
        if re.search(pattern, normalized):
            return "Critical threat pattern detected"
    return None

# ---------------- KEYWORD SCORING ----------------
MALWARE_KEYWORDS = ['virus', 'malware', 'ransomware', 'trojan', 'worm', 'keylogger']
ATTACK_KEYWORDS = ['hack', 'crack', 'breach', 'exploit', 'attack', 'ddos']
SYSTEM_KEYWORDS = ['system', 'server', 'database', 'network', 'admin', 'root']

SAFE_CONTEXT_INDICATORS = [
    'study', 'learn', 'research', 'educational',
    'prevent', 'protect', 'defense', 'security'
]

def enhanced_keyword_scoring(prompt):
    if not prompt:
        return 0.0
    normalized = preprocess_text(prompt)
    words = normalized.split()
    total_words = max(len(words), 1)
    score = 0.0

    score += (sum(1 for w in MALWARE_KEYWORDS if w in normalized) / total_words) * 0.4
    score += (sum(1 for w in ATTACK_KEYWORDS if w in normalized) / total_words) * 0.3
    score += (sum(1 for w in SYSTEM_KEYWORDS if w in normalized) / total_words) * 0.2

    safe_count = sum(1 for w in SAFE_CONTEXT_INDICATORS if w in normalized)
    if safe_count > 0:
        score *= 0.5

    return min(score, 1.0)

# ---------------- ML PREDICTION ----------------
def ml_classifier_predict(prompt):
    if not prompt:
        return 0.0
    try:
        if ml_model is None or tokenizer is None:
            return enhanced_keyword_scoring(prompt)

        processed = preprocess_text(prompt)
        inputs = tokenizer(processed, return_tensors="pt", truncation=True, padding=True, max_length=256)

        with torch.no_grad():
            outputs = ml_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            threat_prob = probs[0][1].item()

        return threat_prob
    except:
        return enhanced_keyword_scoring(prompt)

# ---------------- ANALYSIS ----------------
def analyze_prompt(user_input):
    if not user_input:
        return {'blocked': False, 'response': "Provide a prompt.", 'ml_confidence': 0.0}

    critical = is_critical_threat(user_input)
    if critical:
        return {'blocked': True, 'response': "Malicious content detected.", 'ml_confidence': 1.0}

    ml_score = ml_classifier_predict(user_input)

    if ml_score > 0.8:
        return {'blocked': True, 'response': "Blocked due to malicious content.", 'ml_confidence': ml_score}

    return {'blocked': False, 'response': "Prompt is safe.", 'ml_confidence': ml_score}

# ---------------- ROUTES ----------------
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
        'status': 'healthy',
        'model_loaded': ml_model is not None,
        'version': '2.1'
    })

# ---------------- PRODUCTION START ----------------
if __name__ == '__main__':
    print("🚀 Starting PromptDefender Pro (Production Mode)")
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False
    )
