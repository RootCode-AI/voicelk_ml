import uuid
from datetime import datetime
from flask import Flask, jsonify, request
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import text
from db_config import get_db_engine

app = Flask(__name__)
db_engine = get_db_engine()

@app.route('/', methods=['GET'])
def health_check():
    """Verify database connectivity and backend status."""
    if db_engine:
        return jsonify({"status": "success", "message": "Backend is running!"}), 200
    return jsonify({"status": "error", "message": "Database connection failed."}), 500

@app.route('/api/register', methods=['POST'])
def register_user():
    """Handles new user registration and populates base tables."""
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        role = data.get('role', 'Student')

        if not all([username, email, password]):
            return jsonify({"status": "error", "message": "Missing required fields!"}), 400

        hashed_password = generate_password_hash(password)

        with db_engine.connect() as conn:
            user_insert_query = text("INSERT INTO USER (Role) VALUES (:role)")
            result = conn.execute(user_insert_query, {"role": role})
            user_id = result.lastrowid
            
            reg_user_query = text("""
                INSERT INTO REGISTERED_USER 
                (User_ID, Email, User_Name, Password_Hash, Account_Status) 
                VALUES (:uid, :email, :uname, :pwd, 'Active')
            """)
            conn.execute(reg_user_query, {
                "uid": user_id, "email": email, "uname": username, "pwd": hashed_password
            })
            conn.commit()

        return jsonify({"status": "success", "message": f"User '{username}' registered successfully!"}), 201

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login_user():
    """Authenticates credentials and returns user details on success."""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({"status": "error", "message": "Email and password are required!"}), 400

        with db_engine.connect() as conn:
            query = text("SELECT User_ID, User_Name, Password_Hash, Account_Status FROM REGISTERED_USER WHERE Email = :email")
            result = conn.execute(query, {"email": email}).fetchone()

            if result:
                stored_password_hash = result[2] 
                user_name = result[1]
                
                if check_password_hash(stored_password_hash, password):
                    return jsonify({"status": "success", "message": f"Welcome back, {user_name}!", "user_id": result[0]}), 200
                else:
                    return jsonify({"status": "error", "message": "Invalid password!"}), 401
            else:
                return jsonify({"status": "error", "message": "User not found!"}), 404

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/guest/session', methods=['POST'])
def create_guest_session():
    """Generates a temporary session token and tracks the guest IP."""
    try:
        session_id = str(uuid.uuid4())
        ip_address = request.remote_addr or '127.0.0.1'

        with db_engine.connect() as conn:
            user_insert_query = text("INSERT INTO USER (Role) VALUES ('Guest')")
            result = conn.execute(user_insert_query)
            user_id = result.lastrowid
            
            guest_insert_query = text("""
                INSERT INTO GUEST 
                (User_ID, Session_ID, IP_Address) 
                VALUES (:uid, :sid, :ip)
            """)
            conn.execute(guest_insert_query, {
                "uid": user_id, "sid": session_id, "ip": ip_address
            })
            conn.commit()

        return jsonify({
            "status": "success", 
            "message": "Guest session created successfully!",
            "session_id": session_id,
            "ip_address": ip_address,
            "user_id": user_id
        }), 201

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/query', methods=['POST'])
def submit_query():
    """Processes an incoming Sinhala text query and logs it for TTS generation."""
    try:
        data = request.get_json()
        input_text = data.get('input_text')
        syllabus_topic = data.get('syllabus_topic', 'Uncategorized')
        user_id = data.get('user_id') 

        if not input_text:
            return jsonify({"status": "error", "message": "Input text is required!"}), 400

        current_timestamp = datetime.now()

        with db_engine.connect() as conn:
            query_insert = text("""
                INSERT INTO query (Input_Text, Syllabus_Topic, Timestamp, User_ID) 
                VALUES (:txt, :topic, :ts, :uid)
            """)
            result = conn.execute(query_insert, {
                "txt": input_text,
                "topic": syllabus_topic,
                "ts": current_timestamp,
                "uid": user_id
            })
            conn.commit()
            query_id = result.lastrowid

        return jsonify({
            "status": "success",
            "message": "Query saved successfully, ready for TTS processing.",
            "query_id": query_id
        }), 201

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)