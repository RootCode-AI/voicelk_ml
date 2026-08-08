import sys
import os

from flask import Flask, jsonify, request
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Path resolution: allow imports from model_engine/ at the repo root
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "model_engine"))

from pipeline import TextProcessingPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# Initialise the NLP pipeline once at startup (expensive — tokenizer + lexicon)
_pipeline = TextProcessingPipeline()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health_check():
    """Liveness probe for the VoiceLK ML microservice."""
    return jsonify({
        "status": "healthy",
        "service": "VoiceLK ML Service",
    }), 200


@app.route("/api/generate-ipa", methods=["POST"])
def generate_ipa():
    """
    Accepts mixed Sinhala/English text and returns the IPA phoneme sequence.

    Request body (JSON):
        { "text": "<mixed language string>" }

    Response (JSON):
        {
            "status": "success",
            "raw_input": "...",
            "normalized_text": "...",
            "ipa_sequence": "..."
        }
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({
            "status": "error",
            "message": "'text' field is required and must not be empty.",
        }), 400

    try:
        result = _pipeline.process(text)
        return jsonify({
            "status": "success",
            "raw_input": result["raw_input"],
            "normalized_text": result["normalized_text"],
            "ipa_sequence": result["ipa_sequence"],
        }), 200
    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": str(exc),
        }), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
