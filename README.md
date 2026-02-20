# Prompt Defender – AI-Based Jailbreak & Prompt Injection Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3+-green?style=flat-square&logo=flask)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

## 🔐 Overview

Prompt Defender is a **multi-layered AI security framework** designed to detect prompt injection attacks, jailbreak attempts, and zero-day vulnerabilities in Large Language Models (LLMs).

Built with a **hybrid detection engine** combining transformer-based ML classification with rule-based threat analysis for high accuracy and low false positives.

---

## 🖥️ Interface Preview

> Multi-page security dashboard with real-time threat monitoring, security logs, user profiling, and configurable settings.

| Dashboard | Threat Radar | Security Logs |
|-----------|-------------|---------------|
| Prompt analysis & live stats | Real-time visual threat radar | Filterable event log table |

---

## 🧠 System Architecture

The system integrates **4 detection layers**:

1. **Critical Threat Detection** — Regex-based pattern matching for known attack signatures
2. **Enhanced ML Classifier** — DistilBERT transformer with keyword scoring fusion
3. **Rule Override Engine** — Explicit phrase blocklist with leet-speak normalization
4. **Anomaly Detection Layer** — Statistical analysis of token patterns and obfuscation

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Detection Accuracy | 94.7% |
| False Positive Rate | 0% |
| Average Response Time | 44ms |
| Precision | 100% |
| F1 Score | 93.3% |

---

## ⚙️ Tech Stack

- **Backend:** Python, Flask
- **ML:** PyTorch, HuggingFace Transformers (DistilBERT)
- **Frontend:** HTML5, CSS3, JavaScript, Chart.js
- **Security:** Multi-layer hybrid detection engine

---

## 🚀 Features

- ✅ Real-time prompt analysis via REST API
- ✅ Context-aware educational query detection (safe context recognition)
- ✅ Leet-speak & obfuscation normalization
- ✅ 5-page dashboard (Dashboard, Threat Radar, Logs, User Profiling, Settings)
- ✅ Security event logging with CSV export
- ✅ Dynamic threat scoring with configurable thresholds
- ✅ Graceful fallback — runs in rule-based mode without ML model

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/prompt-defender-ai-security.git
cd prompt-defender-ai-security

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze a prompt for threats |
| `/api/stats` | GET | Get system statistics |
| `/api/logs` | GET | Get security event logs |
| `/api/clear-logs` | POST | Clear all logs |
| `/api/settings` | POST | Save system settings |
| `/health` | GET | System health check |

### Example API Call

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How do I secure my server against SQL injection?"}'
```

---

## 🤖 ML Model

> **Note:** The trained DistilBERT model is not included in this repo due to file size constraints.

The system runs in **rule-based mode** by default (no model required). To enable full ML classification:

1. Place your trained model in `./enhanced_results/`
2. Or set the `MODEL_DIR` environment variable to your model path
3. Compatible with any HuggingFace `AutoModelForSequenceClassification` model

---

## 📁 Project Structure

```
prompt-defender-ai-security/
│
├── app.py                  # Flask app + detection engine
├── requirements.txt        # Python dependencies
│
├── templates/              # Jinja2 HTML templates
│   ├── base.html           # Shared layout
│   ├── dashboard.html      # Main dashboard
│   ├── threat_radar.html   # Real-time radar
│   ├── security_logs.html  # Event log viewer
│   ├── user_profiling.html # Behavior analysis
│   ├── settings.html       # System settings
│   ├── 404.html            # Error page
│   └── 500.html            # Error page
│
└── static/                 # CSS & JavaScript
    ├── style.css
    └── main.js
```

---

## 📌 Future Enhancements

- [ ] Multi-language prompt analysis support
- [ ] Federated model training pipeline
- [ ] Advanced behavioral analytics & user risk scoring
- [ ] Webhook integrations for real-time alerts
- [ ] Docker containerization

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

*Built to demonstrate practical AI security engineering — hybrid ML + rule-based threat detection.*
