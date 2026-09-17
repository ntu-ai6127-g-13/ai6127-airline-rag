import base64
import hashlib
import hmac
import json
import time
import bcrypt
from .config import USERS


def password_ok(username, password, annotators):
    value = annotators.get(username, {})
    pw_hash = value.get('pw_hash', '')
    if username not in USERS or not pw_hash or len(password.encode()) > 72:
        return False
    try:
        return bcrypt.checkpw(password.encode(), pw_hash.encode())
    except ValueError:
        return False


def issue_token(username, annotators):
    payload = base64.urlsafe_b64encode(json.dumps({'u': username, 'exp': int(time.time()) + 43200}).encode()).decode()
    sig = hmac.new(annotators[username]['pw_hash'].encode(), payload.encode(), hashlib.sha256).hexdigest()
    return payload + '.' + sig


def verify_token(token, annotators):
    try:
        payload, sig = token.split('.')
        data = json.loads(base64.urlsafe_b64decode(payload))
        user = data['u']
        expected = hmac.new(annotators[user]['pw_hash'].encode(), payload.encode(), hashlib.sha256).hexdigest()
        if user in USERS and data['exp'] > time.time() and hmac.compare_digest(sig, expected):
            return user
    except (ValueError, KeyError, TypeError, UnicodeError):
        pass
    return None
