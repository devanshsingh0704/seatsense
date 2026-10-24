"""Sign-in for the dashboard: accounts, password checks, lockout, sessions, access log.

Passwords are kept only as salted scrypt hashes (users.json, never in git). A session is a random
token in an HttpOnly, Secure, SameSite=Strict cookie, valid 8 hours; sessions live in memory, so a
restart signs everyone out. Five wrong passwords lock that account for 15 minutes. Camera access is
per account and is enforced by gate(), on the server, not only hidden on the page.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from datetime import datetime

SESSION_S = 8 * 3600
LOCK_AFTER, LOCK_S = 5, 15 * 60
COOKIE = "dp_session"
NAME = re.compile(r"^[A-Za-z0-9._-]{1,32}$")
SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 32}


def hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    return f"scrypt${salt.hex()}${hashlib.scrypt(password.encode(), salt=salt, **SCRYPT).hex()}"


def check_password(password, stored):
    _, salt, digest = stored.split("$")
    return hmac.compare_digest(hash_password(password, bytes.fromhex(salt)).split("$")[2], digest)


DUMMY = hash_password(secrets.token_hex(16))


def cookie_token(header):
    for part in (header or "").split(";"):
        k, _, v = part.strip().partition("=")
        if k == COOKIE and v:
            return v
    return None


def gate(path, user, service=False):
    """What a request may do: "ok", "login" (send the browser to the sign-in page), 401 or 403."""
    if path in ("/login", "/logout"):
        return "ok"
    if path == "/state" and service:            # the frame logger on the same machine
        return "ok"
    if user is None:
        return "login" if path in ("/", "/reports") else 401
    if path.startswith("/frame/") and not user["camera"]:
        return 403
    return "ok"


class Auth:
    def __init__(self, users_path, log_path, service_key):
        self.users_path, self.log_path, self.service_key = users_path, log_path, service_key
        self.users, self.mtime = {}, None
        self.sessions, self.fails = {}, {}
        self._refresh()

    def _refresh(self):
        """Re-read users.json when tools/users.py has changed it, so no restart is needed."""
        m = self.users_path.stat().st_mtime_ns if self.users_path.exists() else None
        if m != self.mtime:
            self.users, self.mtime = (json.loads(self.users_path.read_text()) if m else {}), m

    # --- accounts (tools/users.py) ---
    def _save(self):
        self.users_path.write_text(json.dumps(self.users, indent=1))
        os.chmod(self.users_path, 0o600)
        self.mtime = self.users_path.stat().st_mtime_ns

    def set_user(self, name, password, camera=False):
        if not NAME.match(name):
            raise ValueError("name: 1-32 letters, digits, . _ -")
        if len(password) < 10:
            raise ValueError("password: at least 10 characters")
        self.users[name] = {"hash": hash_password(password), "camera": bool(camera)}
        self._save()

    def set_camera(self, name, camera):
        self.users[name]["camera"] = bool(camera)
        self._save()

    def remove_user(self, name):
        self.users.pop(name, None)
        self.sessions = {t: s for t, s in self.sessions.items() if s["name"] != name}
        self._save()

    # --- sign-in ---
    def _log(self, ip, name, event):
        name = name if NAME.match(name or "") else "?"
        with self.log_path.open("a") as fh:
            fh.write(f"{datetime.now().isoformat(timespec='seconds')} {ip} {name} {event}\n")
        os.chmod(self.log_path, 0o600)   # who signed in when: owner only, like users.json

    def locked(self, name, now=None):
        now = time.time() if now is None else now
        return len([t for t in self.fails.get(name, []) if now - t < LOCK_S]) >= LOCK_AFTER

    def login(self, name, password, ip, now=None):
        now = time.time() if now is None else now
        self._refresh()
        if self.locked(name, now):
            self._log(ip, name, "login refused (locked)")
            return None
        u = self.users.get(name)
        ok = check_password(password, u["hash"] if u else DUMMY)   # same work for unknown names: no timing hint
        if u is None or not ok:
            if u is not None:   # only real accounts collect failures (no memory growth from made-up names)
                self.fails[name] = [t for t in self.fails.get(name, []) if now - t < LOCK_S] + [now]
            self._log(ip, name, "login failed" + (" -> locked 15 min" if self.locked(name, now) else ""))
            return None
        self.fails.pop(name, None)
        token = secrets.token_urlsafe(32)
        self.sessions[token] = {"name": name, "expires": now + SESSION_S}
        self._log(ip, name, "login ok")
        return token

    def user(self, token, now=None):
        now = time.time() if now is None else now
        self._refresh()
        s = self.sessions.get(token or "")
        if s is None or s["expires"] < now or s["name"] not in self.users:
            return None
        return {"name": s["name"], "camera": self.users[s["name"]]["camera"]}

    def logout(self, token, ip):
        s = self.sessions.pop(token or "", None)
        if s:
            self._log(ip, s["name"], "logout")

    def service(self, key):
        return bool(key) and bool(self.service_key) and hmac.compare_digest(key, self.service_key)
