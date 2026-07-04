import re

EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
SINHALA_OR_ENGLISH_PATTERN = re.compile(r"[\u0D80-\u0DFFa-zA-Z]")

GUEST_MAX_INPUT_LENGTH = 500
REGISTERED_MAX_INPUT_LENGTH = 2000


def validate_email(email):
    if not email or not EMAIL_PATTERN.match(email):
        return False, "Invalid email format."
    return True, None


def validate_password(password, confirm_password=None):
    if not password:
        return False, "Password is required."
    if confirm_password is not None and password != confirm_password:
        return False, "Password and confirm password do not match."
    if len(password) <= 10:
        return False, "Password must be more than 10 characters."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number."
    if not re.search(r"[^A-Za-z0-9]", password):
        return False, "Password must contain at least one special character."
    return True, None


def validate_input_text(text, role="Guest"):
    if not text or not text.strip():
        return False, "Input text is required."
    if not SINHALA_OR_ENGLISH_PATTERN.search(text):
        return False, "Input must contain at least one valid Sinhala or English character."
    max_length = (
        REGISTERED_MAX_INPUT_LENGTH
        if role == "REGISTERED"
        else GUEST_MAX_INPUT_LENGTH
    )
    if len(text) > max_length:
        return False, f"Input exceeds the maximum length of {max_length} characters."
    return True, None
