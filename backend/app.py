from flask import Flask, jsonify, request
from db_config import get_db_engine
from sqlalchemy import text
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize Flask App
app = Flask(__name__)

# Initialize Database Engine
db_engine = get_db_engine()

@app.route('/', methods=['GET'])
def health_check():
    """Basic health check route."""
    if db_engine:
        return jsonify({"status": "success", "message": "Backend is running!"}), 200
    return jsonify({"status": "error", "message": "Database connection failed."}), 500

@app.route('/api/register', methods=['POST'])
def register_user():
    """API Endpoint to register a new user."""
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        role = data.get('role', 'Student')

        if not username or not email or not password:
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

# ---------------------------------------------------------
# NEW: Login API
# ---------------------------------------------------------
@app.route('/api/login', methods=['POST'])
def login_user():
    """
    API Endpoint for user login.
    Checks if email exists and verifies the hashed password.
    """
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({"status": "error", "message": "Email and password are required!"}), 400

        with db_engine.connect() as conn:
            # Check if user exists by email
            query = text("SELECT User_ID, User_Name, Password_Hash, Account_Status FROM REGISTERED_USER WHERE Email = :email")
            result = conn.execute(query, {"email": email}).fetchone()

            if result:
                # result contains: (User_ID, User_Name, Password_Hash, Account_Status)
                stored_password_hash = result[2] 
                user_name = result[1]
                
                # Verify password securely
                if check_password_hash(stored_password_hash, password):
                    return jsonify({
                        "status": "success", 
                        "message": f"Welcome back, {user_name}!",
                        "user_id": result[0]
                    }), 200
                else:
                    return jsonify({"status": "error", "message": "Invalid password!"}), 401
            else:
                return jsonify({"status": "error", "message": "User not found!"}), 404

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask development server
    app.run(debug=True, host='127.0.0.1', port=5000)