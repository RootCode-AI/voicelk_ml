# Adaptive Sinhala Text-to-Speech (TTS) System

A dynamic, adaptive Text-to-Speech engine tailored for the Sinhala language. This project provides a robust backend API for processing Sinhala text queries, generating audio via a machine learning model engine, and managing user access through an authentication system.

## 🚀 Current Features (Backend Phase 1)

* **Standardized Project Structure:** Segregated directories for frontend, backend, and model engines.
* **Database Integration:** Configured securely with MySQL and SQLAlchemy.
* **User Authentication:**
  * Registration API with secure password hashing (`Werkzeug`).
  * Login API with credential verification.
* **Guest Sessions:** Seamless API to generate temporary 24-hour UUID tokens and track IP addresses for unregistered users.

## 🛠️ Technology Stack

* **Backend Framework:** Python / Flask
* **Database:** MySQL
* **ORM:** SQLAlchemy
* **Security:** Werkzeug Security (Password Hashing)

## 📁 Folder Structure

```text
adaptive-sinhala-tts/
├── backend/          # Flask APIs and Database logic
│   ├── app.py        # Main backend entry point
│   └── db_config.py  # Database connection setup
├── database/         # SQL scripts and schema designs
├── frontend/         # UI/UX design components
├── model_engine/     # AI/ML TTS models and text processing
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

## 📡 API Endpoints (Available)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check to verify backend and DB connection. |
| POST | `/api/register` | Registers a new user. Requires `username`, `email`, `password`. |
| POST | `/api/login` | Authenticates a user. Requires `email`, `password`. |
| POST | `/api/guest/session` | Creates a temporary guest session. Returns `session_id`. |

> **Note:** Development is actively ongoing. Text Query processing and TTS Model Engine integration are scheduled for the next phase.