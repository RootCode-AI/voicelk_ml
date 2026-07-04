# Adaptive Sinhala Text-to-Speech (TTS) System

A dynamic, adaptive Text-to-Speech engine tailored for the Sinhala language. This project provides a robust backend API for processing Sinhala text queries, generating audio via a machine learning model engine, and managing user access through an authentication system. It specifically targets the Sri Lankan G.C.E. O/L ICT curriculum, handling complex Sinhala-English code-switching natively.

## 🚀 Current Features

### 1. Linguistic Processing & G2P Engine (Active)

* **Text Normalization Pipeline (`normalizer.py`):** Cleans mixed Sinhala-English input, expands ICT operators (`==`, `>=`, `&&`), numbers, percentages, URLs, and IP addresses.
* **Hybrid Code-Switching Handler (`g2p.py`):** Detects language boundaries and routes segments to Sinhala (`sinling`) and English (`eng_to_ipa`) G2P modules.
* **Domain-Specific Master Lexicon:** A structured JSON dictionary (`lexicon.json`) mapped from Grade 10 & 11 ICT textbooks.
* **End-to-End NLP Pipeline (`pipeline.py`):** Chains normalization and G2P into a unified IPA sequence for the VITS acoustic model.
* **Backend NLP API:** `POST /api/nlp/process` and `POST /api/tts/process` expose the pipeline via Flask.

### 2. Backend & Security (Phase 1)

* **Standardized Project Structure:** Segregated directories for frontend, backend, model engine, and database.
* **Database Integration:** MySQL schema in `database/schema.sql` aligned with the SDS EER model.
* **User Authentication:** Registration with SRS password rules, login with account locking after 3 failed attempts.
* **Guest Sessions:** UUID-based sessions with 24-hour expiry validation.

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
│   ├── db_config.py  # Database connection setup
│   ├── nlp_service.py # Bridge to model_engine pipeline
│   └── validators.py # SRS input/password validation
├── database/         # SQL schema (SDS-aligned)
│   └── schema.sql
├── frontend/         # UI/UX (planned)
├── model_engine/     # NLP pipeline and lexicon
│   ├── pipeline.py   # End-to-end normalization + G2P
│   ├── normalizer.py # Stage 1 text normalization
│   ├── g2p.py        # Stage 2 code-switched G2P
│   └── lexicon.json  # O/L ICT pronunciation lexicon
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

Copy `.env.example` to `.env` and set your MySQL credentials.

### 4. Initialize the database

```bash
mysql -u root -p < database/schema.sql
```

### 5. Run the application

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
| GET | `/` | Health check (database + pipeline status). |
| POST | `/api/register` | Registers a new user (SRS password rules). |
| POST | `/api/login` | Authenticates a user (account lock after 3 failures). |
| POST | `/api/guest/session` | Creates a 24-hour guest session. |
| POST | `/api/guest/session/validate` | Validates an active guest session. |
| POST | `/api/nlp/process` | Runs normalization + G2P (no DB write). |
| POST | `/api/tts/process` | Full processing layer + saves QUERY/ANSWER. |
| POST | `/api/query` | Alias for `/api/tts/process`. |

> **Note:** VITS acoustic synthesis (waveform generation) is the next research phase. The NLP pipeline (text → IPA) is active and integrated with the backend.