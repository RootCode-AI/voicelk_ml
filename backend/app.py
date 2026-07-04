import os
import uuid
from datetime import datetime, timedelta

from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import text
from werkzeug.security import check_password_hash, generate_password_hash

from db_config import get_db_engine
from nlp_service import process_text
from validators import validate_email, validate_input_text, validate_password

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
CORS(app)

db_engine = get_db_engine()
GUEST_SESSION_HOURS = int(os.getenv("GUEST_SESSION_HOURS", "24"))
LOCK_DURATION_HOURS = 1
MAX_FAILED_LOGINS = 3


def _db_unavailable():
    return jsonify({"status": "error", "message": "Database connection failed."}), 500


def _get_user_role(user_id):
    if not user_id:
        return "Guest"
    with db_engine.connect() as conn:
        row = conn.execute(
            text("SELECT Role FROM USER WHERE User_ID = :uid"),
            {"uid": user_id},
        ).fetchone()
    return row[0] if row else "Guest"


def _validate_guest_session(session_id):
    with db_engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT g.User_ID, g.Created_At
                FROM GUEST g
                WHERE g.Session_ID = :sid
            """),
            {"sid": session_id},
        ).fetchone()
    if not row:
        return False, "Invalid guest session."
    created_at = row[1]
    if created_at and datetime.now() > created_at + timedelta(hours=GUEST_SESSION_HOURS):
        return False, "Guest session has expired. Please create a new session."
    return True, None


@app.route("/", methods=["GET"])
def health_check():
    if db_engine:
        return jsonify({
            "status": "success",
            "message": "Adaptive Sinhala TTS backend is running.",
            "components": {
                "database": "connected",
                "nlp_pipeline": "available",
                "vits_acoustic_model": "pending",
            },
        }), 200
    return jsonify({"status": "error", "message": "Database connection failed."}), 500


@app.route("/api/register", methods=["POST"])
def register_user():
    if not db_engine:
        return _db_unavailable()
    try:
        data = request.get_json() or {}
        username = (data.get("username") or "").strip()
        email = (data.get("email") or "").strip()
        password = data.get("password")
        confirm_password = data.get("confirm_password", password)

        if not all([username, email, password]):
            return jsonify({"status": "error", "message": "Missing required fields."}), 400

        valid, message = validate_email(email)
        if not valid:
            return jsonify({"status": "error", "message": message}), 400

        valid, message = validate_password(password, confirm_password)
        if not valid:
            return jsonify({"status": "error", "message": message}), 400

        with db_engine.connect() as conn:
            existing = conn.execute(
                text("""
                    SELECT Email, User_Name
                    FROM REGISTERED_USER
                    WHERE Email = :email OR User_Name = :uname
                """),
                {"email": email, "uname": username},
            ).fetchone()
            if existing:
                return jsonify({
                    "status": "error",
                    "message": "Email or username already exists.",
                }), 409

            result = conn.execute(
                text("INSERT INTO USER (Role) VALUES ('REGISTERED')")
            )
            user_id = result.lastrowid
            conn.execute(
                text("""
                    INSERT INTO REGISTERED_USER
                    (User_ID, Email, User_Name, Password_Hash, Account_Status)
                    VALUES (:uid, :email, :uname, :pwd, 'Active')
                """),
                {
                    "uid": user_id,
                    "email": email,
                    "uname": username,
                    "pwd": generate_password_hash(password),
                },
            )
            conn.commit()

        return jsonify({
            "status": "success",
            "message": f"User '{username}' registered successfully.",
            "user_id": user_id,
        }), 201
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/login", methods=["POST"])
def login_user():
    if not db_engine:
        return _db_unavailable()
    try:
        data = request.get_json() or {}
        email = (data.get("email") or "").strip()
        password = data.get("password")

        if not email or not password:
            return jsonify({
                "status": "error",
                "message": "Email and password are required.",
            }), 400

        with db_engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT User_ID, User_Name, Password_Hash, Account_Status,
                           Failed_Login_Count, Lock_Timestamp
                    FROM REGISTERED_USER
                    WHERE Email = :email
                """),
                {"email": email},
            ).fetchone()

            if not result:
                return jsonify({"status": "error", "message": "User not found."}), 404

            user_id, user_name, stored_hash, status, failed_count, lock_ts = result

            if status == "Locked" and lock_ts:
                if datetime.now() < lock_ts + timedelta(hours=LOCK_DURATION_HOURS):
                    return jsonify({
                        "status": "error",
                        "message": "Account is locked. Try again later.",
                    }), 403
                conn.execute(
                    text("""
                        UPDATE REGISTERED_USER
                        SET Account_Status = 'Active',
                            Failed_Login_Count = 0,
                            Lock_Timestamp = NULL
                        WHERE User_ID = :uid
                    """),
                    {"uid": user_id},
                )
                conn.commit()

            if check_password_hash(stored_hash, password):
                conn.execute(
                    text("""
                        UPDATE REGISTERED_USER
                        SET Failed_Login_Count = 0, Lock_Timestamp = NULL,
                            Account_Status = 'Active'
                        WHERE User_ID = :uid
                    """),
                    {"uid": user_id},
                )
                conn.commit()
                return jsonify({
                    "status": "success",
                    "message": f"Welcome back, {user_name}!",
                    "user_id": user_id,
                    "role": "REGISTERED",
                }), 200

            failed_count = (failed_count or 0) + 1
            if failed_count >= MAX_FAILED_LOGINS:
                conn.execute(
                    text("""
                        UPDATE REGISTERED_USER
                        SET Failed_Login_Count = :count,
                            Account_Status = 'Locked',
                            Lock_Timestamp = :lock_ts
                        WHERE User_ID = :uid
                    """),
                    {
                        "count": failed_count,
                        "lock_ts": datetime.now(),
                        "uid": user_id,
                    },
                )
            else:
                conn.execute(
                    text("""
                        UPDATE REGISTERED_USER
                        SET Failed_Login_Count = :count
                        WHERE User_ID = :uid
                    """),
                    {"count": failed_count, "uid": user_id},
                )
            conn.commit()
            return jsonify({"status": "error", "message": "Invalid password."}), 401
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/guest/session", methods=["POST"])
def create_guest_session():
    if not db_engine:
        return _db_unavailable()
    try:
        session_id = str(uuid.uuid4())
        ip_address = request.remote_addr or "127.0.0.1"

        with db_engine.connect() as conn:
            result = conn.execute(text("INSERT INTO USER (Role) VALUES ('Guest')"))
            user_id = result.lastrowid
            conn.execute(
                text("""
                    INSERT INTO GUEST (User_ID, Session_ID, IP_Address, Created_At)
                    VALUES (:uid, :sid, :ip, :created_at)
                """),
                {
                    "uid": user_id,
                    "sid": session_id,
                    "ip": ip_address,
                    "created_at": datetime.now(),
                },
            )
            conn.commit()

        return jsonify({
            "status": "success",
            "message": "Guest session created successfully.",
            "session_id": session_id,
            "ip_address": ip_address,
            "user_id": user_id,
            "expires_in_hours": GUEST_SESSION_HOURS,
        }), 201
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/guest/session/validate", methods=["POST"])
def validate_guest_session():
    if not db_engine:
        return _db_unavailable()
    data = request.get_json() or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"status": "error", "message": "session_id is required."}), 400
    valid, message = _validate_guest_session(session_id)
    if not valid:
        return jsonify({"status": "error", "message": message}), 401
    return jsonify({"status": "success", "message": "Guest session is valid."}), 200


@app.route("/api/nlp/process", methods=["POST"])
def nlp_process():
    """Runs Stage 1 (Normalization) + Stage 2 (G2P) without database writes."""
    try:
        data = request.get_json() or {}
        input_text = data.get("input_text", "")
        user_id = data.get("user_id")
        session_id = data.get("session_id")

        if session_id:
            valid, message = _validate_guest_session(session_id)
            if not valid:
                return jsonify({"status": "error", "message": message}), 401

        role = _get_user_role(user_id) if db_engine and user_id else "Guest"
        valid, message = validate_input_text(input_text, role)
        if not valid:
            return jsonify({"status": "error", "message": message}), 400

        result = process_text(input_text)
        return jsonify({"status": "success", **result}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/tts/process", methods=["POST"])
def tts_process():
    """
    Processing Layer entry point (SDS Section 6.2).
    Currently executes NLP stages; VITS acoustic synthesis is pending.
    """
    try:
        data = request.get_json() or {}
        input_text = data.get("input_text", "")
        user_id = data.get("user_id")
        session_id = data.get("session_id")
        syllabus_topic = data.get("syllabus_topic", "Uncategorized")

        if session_id:
            valid, message = _validate_guest_session(session_id)
            if not valid:
                return jsonify({"status": "error", "message": message}), 401

        role = _get_user_role(user_id) if db_engine and user_id else "Guest"
        valid, message = validate_input_text(input_text, role)
        if not valid:
            return jsonify({"status": "error", "message": message}), 400

        nlp_result = process_text(input_text)
        query_id = None
        answer_id = None

        if db_engine:
            with db_engine.connect() as conn:
                query_result = conn.execute(
                    text("""
                        INSERT INTO QUERY (Input_Text, Syllabus_Topic, Timestamp, User_ID)
                        VALUES (:txt, :topic, :ts, :uid)
                    """),
                    {
                        "txt": input_text,
                        "topic": syllabus_topic,
                        "ts": datetime.now(),
                        "uid": user_id,
                    },
                )
                query_id = query_result.lastrowid
                answer_result = conn.execute(
                    text("""
                        INSERT INTO ANSWER (Response_Text, Source, Query_ID)
                        VALUES (:txt, :source, :qid)
                    """),
                    {
                        "txt": nlp_result["normalized_text"],
                        "source": "NLP Pipeline",
                        "qid": query_id,
                    },
                )
                answer_id = answer_result.lastrowid
                conn.commit()

        return jsonify({
            "status": "success",
            "message": "Text processed. Acoustic synthesis pending VITS integration.",
            "query_id": query_id,
            "answer_id": answer_id,
            "raw_input": nlp_result["raw_input"],
            "normalized_text": nlp_result["normalized_text"],
            "ipa_sequence": nlp_result["ipa_sequence"],
            "audio_status": "pending",
        }), 201
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def submit_query():
    """Legacy endpoint — forwards to the full TTS processing pipeline."""
    return tts_process()


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
