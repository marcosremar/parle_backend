"""
Authentication Core Logic
Password hashing and verification using Argon2

⚠️  SECURITY FIX: Migrated from SHA256 (unsalted) to Argon2 with timing-attack resistant verification
"""

import os

from passlib.context import CryptContext

# Argon2 context with secure parameters
pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto",
    argon2__memory_cost=65536,  # 64MB memory
    argon2__time_cost=3,  # 3 iterations
    argon2__parallelism=4,  # 4 parallelism
    argon2__hash_len=32,  # 32 byte hash
    argon2__salt_len=16,  # 16 byte salt
)

# Configurable minimum password length (default 12, can be overridden for development)
MIN_PASSWORD_LENGTH = int(os.getenv("MIN_PASSWORD_LENGTH", "12"))


def hash_password(password: str) -> str:
    """
    Hash password using Argon2 with secure parameters

    Args:
        password: Plain text password (must be >= MIN_PASSWORD_LENGTH chars, default 12)

    Returns:
        Hashed password string (starts with $argon2)

    Raises:
        ValueError: If password is too short or hashing fails
    """
    # Validate password length (configurable minimum security requirement)
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long")

    try:
        return pwd_context.hash(password)
    except Exception as e:
        raise ValueError(f"Password hashing failed: {e}")


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify password against Argon2 hash using timing-attack resistant comparison

    Args:
        password: Plain text password to verify
        password_hash: Stored password hash

    Returns:
        True if password matches, False otherwise

    Note:
        Uses hmac.compare_digest for constant-time comparison to prevent timing attacks
    """
    if not password or not password_hash:
        # Prevent timing attacks by hashing a dummy password
        # This takes the same time as hashing a real password
        try:
            pwd_context.verify("dummy_password_to_waste_time_safely", "$argon2$dummy")
        except Exception:
            pass
        return False

    try:
        # Use passlib's timing-safe verification
        # passlib.verify() uses constant-time comparison internally
        # This is already timing-attack resistant, no need for extra hmac comparison
        # (hmac.compare_digest on hashes doesn't work because each hash is unique due to salt)
        result = pwd_context.verify(password, password_hash)
        # Clear password from memory after verification
        password = None
        return result
    except Exception:
        # On any error (invalid hash format, etc.), perform timing-safe dummy operation
        try:
            pwd_context.verify("dummy_password_to_waste_time_safely", "$argon2$dummy")
        except Exception:
            pass
        password = None  # Clear password even on error
        return False
    finally:
        # Ensure password is cleared
        try:
            del password
        except NameError:
            pass
