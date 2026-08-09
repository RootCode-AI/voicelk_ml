# voicelk_ml — VoiceLK Machine Learning Service

> **Part of the VoiceLK Adaptive Sinhala TTS System**  
> A dedicated Python AI microservice responsible for NLP text processing and (upcoming) VITS acoustic synthesis. Designed specifically for Sri Lanka's G.C.E. O/L ICT curriculum.

---

## Overview

`voicelk_ml` is the **Python AI microservice** within the larger VoiceLK platform. It operates as a focused, stateless HTTP service that accepts mixed Sinhala-English text and returns a unified IPA (International Phonetic Alphabet) phoneme sequence ready for acoustic model synthesis.

### Platform Architecture

```
┌─────────────────────┐        ┌──────────────────────────┐
│   React Frontend    │◄──────►│    Java Backend (Spring) │
│  (Student/Teacher   │        │  Auth · DB · Sessions    │
│      Portal)        │        │  Business Logic          │
└─────────────────────┘        └────────────┬─────────────┘
                                            │  POST /api/generate-ipa
                                            │  {"text": "..."}
                                            ▼
                                ┌──────────────────────────┐
                                │   voicelk_ml  (THIS)     │
                                │  Python · Flask          │
                                │  NLP Pipeline + VITS     │
                                └──────────────────────────┘
```

This service has **no database**, **no authentication**, and **no session management** — all of that is the responsibility of the Java backend. `voicelk_ml` does one thing: **convert text to phonemes (and eventually audio)**.

---

## Current Features

| Feature | Status |
|---|---|
| Stage 1: Text Normalisation (`normalizer.py`) | ✅ Complete |
| Stage 2: Code-Switched G2P (`g2p.py`) | ✅ Complete |
| Domain Lexicon — O/L ICT Terms (`lexicon.json`) | ✅ Complete |
| End-to-End NLP Pipeline (`pipeline.py`) | ✅ Complete |
| REST API — `/health` & `/api/generate-ipa` | ✅ Complete |
| VITS Acoustic Model (IPA → Audio waveform) | 🔬 Research Phase |

---

## Directory Structure

```
voicelk_ml/
├── api/
│   └── main.py              # Flask microservice entry point (2 endpoints)
│
├── model_engine/            # Core NLP pipeline — do not modify without testing
│   ├── __init__.py          # Exports TextProcessingPipeline
│   ├── pipeline.py          # Chains Stage 1 + Stage 2 into one call
│   ├── normalizer.py        # Stage 1: text cleaning and symbol expansion
│   ├── g2p.py               # Stage 2: code-switched grapheme-to-phoneme
│   └── lexicon.json         # Domain-specific pronunciation dictionary (~83 KB)
│
├── model_training/          # (Future) VITS model training scripts and configs
│
├── data/                    # (Future) Raw and preprocessed training datasets
│
├── .env.example             # Environment variable template
├── .gitignore
└── requirements.txt         # Python dependencies (NLP only)
```

### Folder Explanations

| Folder / File | Purpose |
|---|---|
| `api/main.py` | Lightweight Flask app. Receives requests from the Java backend and returns IPA sequences. |
| `model_engine/` | The complete, production-ready NLP pipeline. **Core logic — handle with care.** |
| `model_engine/normalizer.py` | Cleans raw input: expands symbols (`==` → `සමානයි`), numbers (`1024` → `එක්දහස් විසි හතර`), URLs, IP addresses, percentages. |
| `model_engine/g2p.py` | Detects Sinhala vs. English tokens and routes each to the correct phoneme module. Handles acronyms intelligently (RAM → ɑːr eɪ em). |
| `model_engine/lexicon.json` | Hand-curated pronunciation dictionary for O/L ICT terms in both English and Sinhala. Overrides library defaults for domain accuracy. |
| `model_engine/pipeline.py` | Composes `normalizer.py` → `g2p.py` into a single `process(text)` call. |
| `model_training/` | Will contain training scripts, configs, and checkpoints for the VITS acoustic model. |
| `data/` | Will contain raw audio recordings, transcripts, and preprocessed feature files for model training. |

---

## Architecture & Data Flow

### Text-to-IPA Pipeline (Active)

```
Java Backend sends:
  POST /api/generate-ipa
  {"text": "RAM එක 1024 MB වේ. A == B නම් 100% නිවැරදියි."}

                      │
                      ▼
         ┌──────────────────────────┐
         │  Stage 1: normalizer.py  │
         │                          │
         │  • Expand URL / IP       │  "192.168.1.1" → "එකසිය... ඩොට් ..."
         │  • Expand percentages    │  "100%" → "සියයට සියයක්"
         │  • Expand operators      │  "==" → "සමානයි", "&&" → "සහ"
         │  • Expand numbers        │  "1024" → "එක්දහස් විසි හතර"
         │  • Strip noise/emojis    │
         └───────────┬──────────────┘
                     │  Cleaned Sinhala text
                     ▼
         ┌──────────────────────────┐
         │  Stage 2: g2p.py         │
         │                          │
         │  Sinling tokeniser splits│
         │  text into segments:     │
         │                          │
         │  [English token]         │  → eng_to_ipa / letter-spell / lexicon
         │  [Sinhala token]         │  → consonant-modifier rules / lexicon
         │  [Punctuation]           │  → retained as-is                      
         └───────────┬──────────────┘
                     │  Unified IPA sequence
                     ▼

Java Backend receives:
  {
    "status": "success",
    "raw_input": "RAM එක 1024 MB වේ...",
    "normalized_text": "ɑːr eɪ em එක...",
    "ipa_sequence": "ɑːr eɪ em aka..."
  }
```

### Code-Switching — How It Works

The G2P engine handles native Sinhala-English mixing at the **token level**:

| Token Type | Example | Routing | Output |
|---|---|---|---|
| English word | `Database` | `eng_to_ipa` library | `ˈdeɪtəbeɪs` |
| ALL-CAPS acronym | `RAM` | Letter-spelling | `ɑːr eɪ em` |
| Lexicon word | `HTML` | `lexicon.json` override | custom IPA |
| Sinhala word | `දත්ත` | Rule-based decomposition | `d̪at̪t̪a` |
| Sinhala ICT term | `පරිගණකය` | `si_lexicon` override | precise IPA |
| Punctuation | `.` `,` | Passed through | `.` `,` |

---

## API Reference

### `GET /health`

Liveness probe. Returns service status.

**Response `200`:**
```json
{
  "status": "healthy",
  "service": "VoiceLK ML Service"
}
```

---

### `POST /api/generate-ipa`

Accepts mixed Sinhala-English text and returns the full IPA phoneme sequence.

**Request body:**
```json
{
  "text": "Database Management System එකක් හදන්නේ කෙසේද?"
}
```

**Response `200`:**
```json
{
  "status": "success",
  "raw_input": "Database Management System එකක් හදන්නේ කෙසේද?",
  "normalized_text": "Database Management System ekak hadanne kesed?",
  "ipa_sequence": "ˈdeɪtəbeɪs ˈmænɪdʒmənt ˈsɪstəm ekak hadanne kesed ?"
}
```

**Response `400` — missing or empty text:**
```json
{
  "status": "error",
  "message": "'text' field is required and must not be empty."
}
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/RootCode-AI/adaptive-sinhala-tts.git
cd adaptive-sinhala-tts
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment (optional)

```bash
cp .env.example .env
# Edit .env if needed — no database credentials required for this service
```

### 5. Start the service

```bash
python api/main.py
```

The service starts on `http://127.0.0.1:5000`.

### 6. Test the endpoints

```bash
# Health check
curl http://127.0.0.1:5000/health

# IPA generation
curl -X POST http://127.0.0.1:5000/api/generate-ipa \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"RAM එක 1024 MB වේ.\"}"
```

---

## Technology Stack

| Layer | Technology | Version |
|---|---|---|
| API Framework | Python / Flask | `>=3.0.0` |
| Cross-Origin | Flask-CORS | `>=4.0.0` |
| Sinhala Tokenisation | `sinling` | `>=0.1.0` |
| English Phonetics | `eng-to-ipa` | `>=0.0.2` |
| Environment Config | `python-dotenv` | `>=1.0.0` |
| Acoustic Model *(future)* | VITS Neural TTS | — |

---

## Roadmap

- [x] Stage 1: Text normalisation pipeline
- [x] Stage 2: Code-switched G2P engine
- [x] Domain lexicon — Grade 10 & 11 O/L ICT terms
- [x] REST API microservice
- [ ] VITS acoustic model integration (IPA → waveform)
- [ ] Model training infrastructure (`model_training/`)
- [ ] Sinhala speech corpus compilation (`data/`)
- [ ] Docker containerisation for deployment

---

## Related Repositories

| Service | Language | Responsibility |
|---|---|---|
| `voicelk_ml` *(this)* | Python | NLP Pipeline · Acoustic Model |
| `voicelk_backend` | Java (Spring Boot) | Auth · Database · Business Logic |
| `voicelk_frontend` | React | Student / Teacher UI |

---

> **Note:** This service is stateless. It holds no user data, performs no authentication, and writes to no database. All sensitive operations are handled by the Java backend.