# Security Module - Ultravox Pipeline

Complete security hardening suite for the Ultravox Pipeline.

## Features

### 🔐 JWT Manager
- **Access + Refresh tokens** with automatic rotation
- **Persistent blacklist** (Redis/Database support)
- **HS256 and RS256** algorithms
- **Role-Based Access Control (RBAC)** with fine-grained permissions
- **Brute force protection** with account lockout
- **Token versioning** for forced invalidation

### 🚦 Advanced Rate Limiter
- **Multiple strategies**: Fixed Window, Sliding Window, Token Bucket
- **Multiple backends**: Memory, Redis (distributed)
- **Multiple scopes**: Per-IP, Per-User, Per-API-Key, Per-Endpoint
- **Gradual backoff** instead of hard blocking
- **Whitelist/Blacklist** support
- **Real-time metrics**

### 🛡️ Input Sanitizer
- **XSS protection** (script tags, event handlers, javascript:)
- **SQL injection detection** (UNION, OR 1=1, comments)
- **CSRF token** generation and validation
- **File upload validation** (magic numbers, size limits, path traversal)
- **JSON depth limiting** (DoS prevention)
- **URL sanitization** (scheme validation, private IP blocking)

### 🗝️ Secrets Manager
- **Multiple backends**: Environment, HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager
- **Secret caching** with TTL
- **Encryption at rest** for cached secrets
- **Secret rotation** support
- **Audit logging** of secret access
- **Never logs secrets** in plaintext

### 📊 Security Audit Logger
- **Centralized audit trail** for all security events
- **Immutable logs** with hash verification
- **Multiple backends**: File, Database, Elasticsearch
- **Event categorization** (auth, data access, config changes, security incidents)
- **Query capabilities** (by type, user, time range)
- **Retention policies** with automatic cleanup
- **Compliance support** (GDPR, SOC2, PCI-DSS)

### 🧩 Security Middleware
- **All-in-one middleware** combining all security features
- **HTTPS enforcement**
- **Automatic security headers** (X-Content-Type-Options, X-Frame-Options, HSTS)
- **Request-level audit logging**
- **Easy integration** with FastAPI

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r src/core/security/requirements.txt

# Optional: Redis for distributed features
pip install redis

# Optional: HashiCorp Vault
pip install hvac

# Optional: AWS Secrets Manager
pip install boto3

# Optional: GCP Secret Manager
pip install google-cloud-secret-manager

# Optional: Elasticsearch for audit logs
pip install elasticsearch
```

### 2. Basic Usage

```python
from fastapi import FastAPI
from src.core.security import create_security_middleware

app = FastAPI()

# Add integrated security middleware
middleware = create_security_middleware(
    jwt_config={
        "secret_key": "your-secret-key-at-least-32-characters",
        "algorithm": "HS256",
        "access_token_expire_minutes": 15,
        "refresh_token_expire_days": 7,
    },
    rate_limit_config={
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
        "strategy": "sliding_window",
    },
    sanitization_config={
        "level": "strict",
        "enable_xss_protection": True,
        "enable_sql_injection_detection": True,
        "enable_csrf_protection": True,
    },
    audit_config={
        "backend": "file",
        "log_file_path": "logs/audit.log",
    },
    require_auth=True,
    require_https=True,
    public_paths=["/health", "/docs"],
)

app.add_middleware(middleware)
```

---

## Detailed Usage

### JWT Authentication

```python
from src.core.security import JWTManager, JWTConfig, UserRole, Permission

# Configure JWT
config = JWTConfig(
    secret_key="your-secret-key-at-least-32-characters",
    algorithm="HS256",
    access_token_expire_minutes=15,
    refresh_token_expire_days=7,
    enable_refresh_rotation=True,
    blacklist_backend="redis",
    redis_url="redis://localhost:6379",
)

manager = JWTManager(config)

# Create token pair
tokens = manager.create_token_pair(
    user_id="user123",
    role=UserRole.ADMIN,
    permissions=[Permission.READ_USERS, Permission.WRITE_USERS],
)

print(f"Access token: {tokens.access_token}")
print(f"Refresh token: {tokens.refresh_token}")

# Verify access token
try:
    payload = manager.verify_access_token(
        tokens.access_token,
        required_role=UserRole.USER,  # Minimum role required
    )
    print(f"User ID: {payload['sub']}")
    print(f"Role: {payload['role']}")
except Exception as e:
    print(f"Invalid token: {e}")

# Refresh access token
new_access, new_refresh = manager.refresh_access_token(tokens.refresh_token)

# Revoke token
payload = manager.verify_access_token(tokens.access_token)
manager.revoke_token(payload['jti'])
```

### Rate Limiting

```python
from src.core.security import (
    AdvancedRateLimiter,
    RateLimitConfig,
    RateLimitStrategy,
    RateLimitScope,
)

# Configure rate limiter
config = RateLimitConfig(
    strategy=RateLimitStrategy.SLIDING_WINDOW,
    requests_per_minute=60,
    requests_per_hour=1000,
    enable_gradual_backoff=True,
    whitelist_ips={"192.168.1.100"},
)

limiter = AdvancedRateLimiter(config)

# Check rate limit
result = limiter.check_limit(
    identifier="192.168.1.1",
    scope=RateLimitScope.IP,
    endpoint="/api/v1/users",
)

if not result.allowed:
    print(f"Rate limited! Retry after {result.retry_after}s")
elif result.throttle_delay_ms:
    print(f"Throttling: delay {result.throttle_delay_ms}ms")
    import asyncio
    await asyncio.sleep(result.throttle_delay_ms / 1000)
```

### Input Sanitization

```python
from src.core.security import InputSanitizer, SanitizationConfig

# Configure sanitizer
config = SanitizationConfig(
    level="strict",
    enable_xss_protection=True,
    enable_sql_injection_detection=True,
    enable_csrf_protection=True,
)

sanitizer = InputSanitizer(config)

# Check for XSS
user_input = "<script>alert('xss')</script>Hello"
if sanitizer.detect_xss(user_input):
    print("XSS detected!")

# Sanitize HTML
safe_html = sanitizer.sanitize_html(user_input)
print(f"Safe HTML: {safe_html}")

# Generate CSRF token
csrf_token = sanitizer.generate_csrf_token("user123")

# Validate CSRF token
if sanitizer.validate_csrf_token(csrf_token, "user123"):
    print("CSRF token valid")

# Validate file upload
with open("upload.jpg", "rb") as f:
    valid, error = sanitizer.validate_file_upload(
        filename="upload.jpg",
        content=f.read(),
    )
    if not valid:
        print(f"Invalid file: {error}")
```

### Secrets Management

```python
from src.core.security import SecretsManager, SecretConfig, SecretsBackend

# Environment variables (default)
config = SecretConfig(backend=SecretsBackend.ENVIRONMENT)
manager = SecretsManager(config)
jwt_secret = manager.get_secret("JWT_SECRET_KEY")

# HashiCorp Vault
config = SecretConfig(
    backend=SecretsBackend.VAULT,
    vault_url="https://vault.example.com",
    vault_token="s.xxxxxx",
)
manager = SecretsManager(config)
db_password = manager.get_secret("database/password")

# AWS Secrets Manager
config = SecretConfig(
    backend=SecretsBackend.AWS,
    aws_region="us-east-1",
)
manager = SecretsManager(config)
api_key = manager.get_secret("api/groq_key")

# Set secret (if backend supports it)
manager.set_secret("api/new_key", "secret-value-123")

# Mask secret for logging
masked = manager.mask_secret("my-secret-key-12345")
print(f"API Key: {masked}")  # Output: "API Key: my-s***45"
```

### Security Audit Logging

```python
from src.core.security import (
    SecurityAuditLogger,
    AuditLogConfig,
    AuditEventType,
    AuditSeverity,
)

# Configure audit logger
config = AuditLogConfig(
    backend="file",
    log_file_path="logs/audit.log",
    retention_days=365,
)

logger = SecurityAuditLogger(config)

# Log authentication event
logger.log_event(
    event_type=AuditEventType.AUTH_LOGIN,
    action="User logged in successfully",
    severity=AuditSeverity.INFO,
    user_id="user123",
    ip_address="192.168.1.1",
    trace_id="abc123",
)

# Log security incident
logger.log_event(
    event_type=AuditEventType.SECURITY_XSS_DETECTED,
    action="XSS attack detected",
    severity=AuditSeverity.CRITICAL,
    ip_address="192.168.1.100",
    resource="/api/v1/users",
    success=False,
    metadata={"attack_vector": "<script>alert(1)</script>"},
)

# Query audit logs
events = logger.query_events(
    event_type=AuditEventType.AUTH_FAILED_LOGIN,
    user_id="user123",
    limit=100,
)

for event in events:
    print(f"{event.timestamp}: {event.action}")

# Get statistics
stats = logger.get_statistics()
print(f"Total events: {stats['total_events']}")
print(f"Failed events: {stats['failed_events']}")
```

---

## Testing

Run the security test suite:

```bash
# Run all security tests
pytest src/core/security/tests/ -v

# Run specific test file
pytest src/core/security/tests/test_jwt_manager.py -v

# Run with coverage
pytest src/core/security/tests/ --cov=src/core/security --cov-report=html
```

---

## Security Best Practices

### 1. JWT Tokens
- ✅ Use strong secret keys (minimum 32 characters)
- ✅ Enable HTTPS enforcement
- ✅ Set appropriate token expiration times
- ✅ Use refresh token rotation
- ✅ Implement token blacklist with persistent storage (Redis)
- ✅ Never log tokens in plaintext

### 2. Rate Limiting
- ✅ Use distributed backend (Redis) for multi-instance deployments
- ✅ Set appropriate limits per endpoint
- ✅ Whitelist trusted IPs
- ✅ Monitor rate limit metrics

### 3. Input Validation
- ✅ Enable all protections (XSS, SQL injection, CSRF)
- ✅ Use strict validation level for production
- ✅ Validate file uploads with magic numbers
- ✅ Sanitize all user inputs

### 4. Secrets Management
- ✅ Use HashiCorp Vault or AWS/GCP Secrets Manager in production
- ✅ Never hardcode secrets in code
- ✅ Enable secret rotation
- ✅ Audit all secret access
- ✅ Mask secrets in logs

### 5. Audit Logging
- ✅ Log all security-relevant events
- ✅ Use Elasticsearch for production (searchability)
- ✅ Set appropriate retention policies
- ✅ Monitor critical security events
- ✅ Ensure log immutability

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Security Middleware                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  HTTPS   │→ │   Rate   │→ │  Input   │→ │   JWT    │   │
│  │  Check   │  │  Limit   │  │Sanitize  │  │   Auth   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
         ↓                ↓                ↓                ↓
    ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
    │ Secrets │     │  Redis  │     │  Vault  │     │Elasticsearch│
    │ Manager │     │(Blacklist│    │(Secrets)│    │  (Audit)  │
    └─────────┘     └─────────┘     └─────────┘     └─────────┘
```

---

## Compliance

The security module helps achieve compliance with:

- **GDPR**: Audit logging, data access tracking, retention policies
- **SOC 2**: Centralized audit trail, access controls, encryption
- **PCI-DSS**: Strong authentication, encryption, audit logging
- **OWASP Top 10**: Protection against injection, XSS, authentication issues

---

## Troubleshooting

### Issue: "Secret key must be at least 32 characters"
**Solution**: Generate a strong secret key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Issue: "Redis connection failed"
**Solution**: Ensure Redis is running:
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

### Issue: "Token has been revoked"
**Solution**: Token was explicitly revoked or blacklist has expired tokens. Generate new tokens.

### Issue: "Rate limit exceeded"
**Solution**: Wait for the retry_after duration or reset limits for the identifier.

---

## License

Part of the Ultravox Pipeline project.
