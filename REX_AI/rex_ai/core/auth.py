"""Accounts: sign up, log in, roles (owner: Member 3).

    user    uses the REX Course Assistant; their conversations are stored under their account and only they see them
    admin   uses every agent (RFP, curriculum, documents, marketing, data); everything administrators create is
            stored in the shared tables, so every administrator sees every update

Passwords are stored as PBKDF2-SHA256 hashes with a random salt; a login returns a random token whose SHA-256 is
stored (auth_tokens), valid for REX_LOGIN_DAYS. Signing up as an administrator needs REX_ADMIN_SIGNUP_CODE - except
the very first administrator, so a new installation can be set up.

    python -m rex_ai.core.auth create-admin --email dean@academy.example --name "Dean"
    python -m rex_ai.core.auth set-role --email sara@example.com --role admin
    python -m rex_ai.core.auth list
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from rex_ai import config
from rex_ai.core import db

ROLES = ("user", "admin")
ITERATIONS = 240_000
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD = 8


class AuthError(ValueError):
    pass


def hash_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode(), ITERATIONS).hex()
    return f"pbkdf2_sha256${ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt, digest = stored.split("$")
        check = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode(), int(iterations)).hex()
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(check, digest)


def public(user: dict) -> dict:
    """A user without the password hash."""
    return {k: user.get(k) for k in ("user_id", "email", "name", "role", "created_at", "last_login_at")}


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def signup(name: str, email: str, password: str, role: str = "user", admin_code: Optional[str] = None,
           trusted: bool = False) -> dict:
    """trusted=True: run on the server itself (CLI), no administrator code needed."""
    name, email = (name or "").strip(), (email or "").strip().lower()
    if not name:
        raise AuthError("Enter your name")
    if not EMAIL.match(email):
        raise AuthError("Enter a valid e-mail address")
    if len(password or "") < MIN_PASSWORD:
        raise AuthError(f"The password needs at least {MIN_PASSWORD} characters")
    if role not in ROLES:
        raise AuthError("Role must be user or admin")
    if role == "admin" and not trusted and db.count_users("admin") > 0:
        if not config.ADMIN_SIGNUP_CODE:
            raise AuthError("Administrator sign-up is closed: ask an administrator to give your account the "
                            "administrator role")
        if not hmac.compare_digest((admin_code or "").strip(), config.ADMIN_SIGNUP_CODE):
            raise AuthError("Wrong administrator code")
    if db.get_user(email=email):
        raise AuthError("An account with this e-mail already exists - log in instead")
    user = {"user_id": "USR-" + uuid.uuid4().hex[:10], "email": email, "name": name, "role": role,
            "password_hash": hash_password(password),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    db.create_user(user)
    db.log_event("INFO", "signup", detail=f"{user['user_id']} {email} ({role})")
    return public(user)


def login(email: str, password: str) -> tuple[dict, str]:
    """(user, token). The same message for an unknown e-mail and a wrong password."""
    user = db.get_user(email=(email or "").strip().lower())
    if not user or not verify_password(password or "", user["password_hash"]):
        raise AuthError("Wrong e-mail or password")
    token = secrets.token_urlsafe(32)
    expires = (datetime.now(timezone.utc) + timedelta(days=config.LOGIN_DAYS)).isoformat(timespec="seconds")
    db.save_token(_token_hash(token), user["user_id"], expires)
    db.touch_login(user["user_id"])
    db.log_event("INFO", "login", detail=user["user_id"])
    return public(user), token


def user_for_token(token: Optional[str]) -> Optional[dict]:
    if not token:
        return None
    user = db.user_for_token(_token_hash(token))
    return public(user) if user else None


def logout(token: Optional[str]) -> None:
    if token:
        db.delete_token(_token_hash(token))


def set_role(user_id: str, role: str) -> dict:
    if role not in ROLES:
        raise AuthError("Role must be user or admin")
    user = db.get_user(user_id=user_id)
    if not user:
        raise LookupError(f"User {user_id} not found")
    if user["role"] == "admin" and role == "user" and db.count_users("admin") <= 1:
        raise AuthError("Keep at least one administrator")
    db.set_user_role(user_id, role)
    db.log_event("INFO", "role_changed", detail=f"{user_id}: {user['role']} -> {role}")
    return public({**user, "role": role})


def _main(argv=None) -> None:
    import argparse
    import getpass
    ap = argparse.ArgumentParser(description="REX AI accounts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create-admin", help="create an administrator account (asks for the password)")
    c.add_argument("--email", required=True)
    c.add_argument("--name", required=True)
    r = sub.add_parser("set-role", help="make an account user or admin")
    r.add_argument("--email", required=True)
    r.add_argument("--role", choices=ROLES, required=True)
    sub.add_parser("list")
    args = ap.parse_args(argv)
    db.init_db(seed=False)
    if args.cmd == "create-admin":
        password = getpass.getpass("Password: ")
        user = signup(args.name, args.email, password, "admin", trusted=True)
        print(f"Administrator {user['email']} created ({user['user_id']})")
    elif args.cmd == "set-role":
        user = db.get_user(email=args.email)
        if not user:
            raise SystemExit(f"No account {args.email}")
        print(set_role(user["user_id"], args.role))
    else:
        for u in db.list_users():
            print(f"{u['user_id']}  {u['role']:<5}  {u['email']:<32} {u['name']}  last login {u['last_login_at']}")


if __name__ == "__main__":
    _main()
