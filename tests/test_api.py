"""
Tests for the Flask IPA service (api/main.py), run in-process through Flask's test
client, so no server has to be started first.

    python -m pytest tests
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api"))

from main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "healthy"


def test_generate_ipa_code_switched_sentence(client):
    # A code-switched sentence with an acronym, a lexicon word and a number
    response = client.post("/api/generate-ipa", json={"text": "පරිගණකයේ මොළය CPU එකයි. RAM එක 1024 MB වේ."})
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"
    # The number sits against an English unit, so it is read in English
    assert "one thousand twenty four MB" in body["normalized_text"]
    assert "1024" not in body["normalized_text"]
    assert body["ipa_sequence"]
    # Nothing in the IPA may be left in Sinhala script or as raw digits
    assert not any("඀" <= ch <= "෿" or ch.isdigit() for ch in body["ipa_sequence"])


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {}, None])
def test_generate_ipa_rejects_empty_text(client, payload):
    response = client.post("/api/generate-ipa", json=payload) if payload is not None \
        else client.post("/api/generate-ipa", data="not json", content_type="text/plain")
    assert response.status_code == 400
    assert response.get_json()["status"] == "error"
