"""Simplified auth functions for Nomad"""
from passlib.context import CryptContext

pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto",
    argon2__memory_cost=65536,
    argon2__time_cost=3,
    argon2__parallelism=4,
    argon2__hash_len=32,
    argon2__salt_len=16,
)

def hash_password(password: str) -> str:
    """Hash password using Argon2"""
    if not password or len(password) < 12:
        raise ValueError("Password must be at least 12 characters long")
    return pwd_context.hash(password)

def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against Argon2 hash"""
    if not password or not password_hash:
        return False
    try:
        return pwd_context.verify(password, password_hash)
    except Exception:
        return False

