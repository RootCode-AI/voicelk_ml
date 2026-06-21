# Adaptive Sinhala Text-to-Speech (TTS) System

A dynamic, adaptive Text-to-Speech engine tailored for the Sinhala language. This project provides a robust backend API for processing Sinhala text queries, generating audio via a machine learning model engine, and managing user access through an authentication system. It specifically targets the Sri Lankan G.C.E. O/L ICT curriculum, handling complex Sinhala-English code-switching natively.

## 🚀 Current Features

### 1. Linguistic Processing & G2P Engine (Active)

* **Hybrid Code-Switching Handler:** Accurately detects language boundaries and segments English and Sinhala text using regular expressions.
* **Domain-Specific Master Lexicon:** A highly structured, exhaustive JSON dictionary (`lexicon.json`) mapped directly from Grade 10 & 11 ICT textbooks. It overrides generic text-to-speech engines to ensure accurate Sri Lankan pronunciations of technical jargon, acronyms, and complex Sinhala conjuncts.
* **Unified Phonetic Integration:** Dynamically routes segregated segments to distinct G2P modules (`sinling` for Sinhala, `eng_to_ipa` for English) and merges them into a single, cohesive International Phonetic Alphabet (IPA) sequence for the VITS acoustic model.

### 2. Backend & Security (Phase 1)

* **Standardized Project Structure:** Segregated directories for frontend, backend, and model engines.
* **Database Integration:** Configured securely with MySQL and SQLAlchemy.
* **User Authentication:**
  * Registration API with secure password hashing (`Werkzeug`).
  * Login API with credential verification.
* **Guest Sessions:** Seamless API to generate temporary 24-hour UUID tokens and track IP addresses for unregistered users.

## 🛠️ Technology Stack

* **Backend Framework:** Python / Flask
* **Database:** MySQL / SQLAlchemy
* **Linguistics & NLP:** `sinling` (Sinhala Tokenization), `eng_to_ipa` (English Phonetics), `re` (RegEx Filtering)
* **Security:** Werkzeug Security (Password Hashing)

## 📁 Folder Structure

```text
adaptive-sinhala-tts/
├── backend/          # Flask APIs and Database logic
│   ├── app.py        # Main backend entry point
│   └── db_config.py  # Database connection setup
├── database/         # SQL scripts and schema designs
├── frontend/         # UI/UX design components
├── model_engine/     # AI/ML TTS models and linguistic processing
│   ├── g2p.py        # Advanced Code-Switched G2P Routing Logic
│   └── lexicon.json  # O/L ICT Master Pronunciation Lexicon
├── .env              # Environment variables (Not pushed to version control)
├── .gitignore        # Git ignore rules
└── requirements.txt  # Python dependencies
```

## ⚙️ Local Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/RootCode-AI/adaptive-sinhala-tts.git
cd adaptive-sinhala-tts
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

Ensure your local MySQL server is running, and execute:

```bash
python backend/app.py
```

The server will start on `http://127.0.0.1:5000`

### 5. Test the G2P Engine (Standalone)

You can verify the Code-Switched G2P sequence generation by running:

```bash
python -X utf8 model_engine/g2p.py
```

## 📡 API Endpoints (Available)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check to verify backend and DB connection. |
| POST | `/api/register` | Registers a new user. Requires `username`, `email`, `password`. |
| POST | `/api/login` | Authenticates a user. Requires `email`, `password`. |
| POST | `/api/guest/session` | Creates a temporary guest session. Returns `session_id`. |

> **Note:** The Linguistic Processing module is currently active. The Text Normalization Pipeline (cleaning text, expanding numbers) is actively under development for the next phase.