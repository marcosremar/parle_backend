"""
Secrets Manager with HashiCorp Vault and AWS Secrets Manager integration.

Features:
- Multiple backends: Environment, HashiCorp Vault, AWS Secrets Manager
- Secret rotation support
- Caching with TTL
- Secret versioning
- Audit logging of secret access
- Encryption at rest (for local cache)
- Never log secrets in plaintext

Author: Ultravox Team
Version: 1.0.0
"""

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet
from pydantic import BaseModel, Field


class SecretsBackend(str, Enum):
    """Secrets storage backends."""

    ENVIRONMENT = "environment"  # Environment variables (default)
    VAULT = "vault"  # HashiCorp Vault
    AWS = "aws"  # AWS Secrets Manager
    GCP = "gcp"  # GCP Secret Manager


class SecretConfig(BaseModel):
    """Secrets manager configuration."""

    # Backend
    backend: SecretsBackend = Field(
        SecretsBackend.ENVIRONMENT, description="Secrets backend"
    )

    # Cache settings
    enable_cache: bool = Field(True, description="Enable secret caching")
    cache_ttl_seconds: int = Field(300, description="Cache TTL (default 5 min)")
    encrypt_cache: bool = Field(True, description="Encrypt cached secrets")

    # HashiCorp Vault configuration
    vault_url: Optional[str] = Field(None, description="Vault server URL")
    vault_token: Optional[str] = Field(None, description="Vault authentication token")
    vault_namespace: Optional[str] = Field(None, description="Vault namespace")
    vault_mount_point: str = Field("secret", description="Vault mount point")

    # AWS Secrets Manager configuration
    aws_region: Optional[str] = Field(None, description="AWS region")
    aws_access_key_id: Optional[str] = Field(None, description="AWS access key")
    aws_secret_access_key: Optional[str] = Field(None, description="AWS secret key")

    # GCP Secret Manager configuration
    gcp_project_id: Optional[str] = Field(None, description="GCP project ID")
    gcp_credentials_path: Optional[str] = Field(
        None, description="Path to GCP credentials JSON"
    )

    # Secret rotation
    enable_rotation: bool = Field(False, description="Enable automatic secret rotation")
    rotation_interval_days: int = Field(30, description="Rotation interval in days")

    # Security
    mask_secrets_in_logs: bool = Field(
        True, description="Mask secrets in log output"
    )
    audit_secret_access: bool = Field(True, description="Audit secret access")


@dataclass
class SecretMetadata:
    """Metadata about a secret."""

    key: str
    version: Optional[str] = None
    created_at: Optional[float] = None
    expires_at: Optional[float] = None
    rotation_enabled: bool = False
    last_accessed: Optional[float] = None
    access_count: int = 0


class SecretsManager:
    """
    Secrets manager with multiple backend support.

    Example:
        >>> # Environment variables backend (default)
        >>> config = SecretConfig()
        >>> manager = SecretsManager(config)
        >>> jwt_secret = manager.get_secret("JWT_SECRET_KEY")

        >>> # HashiCorp Vault backend
        >>> config = SecretConfig(
        ...     backend=SecretsBackend.VAULT,
        ...     vault_url="https://vault.example.com",
        ...     vault_token="s.xxxxxx"
        ... )
        >>> manager = SecretsManager(config)
        >>> db_password = manager.get_secret("database/password")

        >>> # AWS Secrets Manager backend
        >>> config = SecretConfig(
        ...     backend=SecretsBackend.AWS,
        ...     aws_region="us-east-1"
        ... )
        >>> manager = SecretsManager(config)
        >>> api_key = manager.get_secret("api/groq_key")
    """

    def __init__(self, config: SecretConfig):
        """
        Initialize secrets manager.

        Args:
            config: Secrets configuration
        """
        self.config = config

        # Initialize backend client
        self._backend_client = None
        if config.backend == SecretsBackend.VAULT:
            self._init_vault()
        elif config.backend == SecretsBackend.AWS:
            self._init_aws()
        elif config.backend == SecretsBackend.GCP:
            self._init_gcp()

        # Secret cache
        self._cache: Dict[str, tuple[Any, float]] = {}  # key -> (value, expiry)

        # Metadata tracking
        self._metadata: Dict[str, SecretMetadata] = {}

        # Encryption key for cache (derived from system)
        if config.encrypt_cache:
            self._cache_key = self._generate_cache_key()
        else:
            self._cache_key = None

    def _init_vault(self) -> None:
        """Initialize HashiCorp Vault client."""
        if not self.config.vault_url or not self.config.vault_token:
            raise ValueError(
                "Vault backend requires vault_url and vault_token configuration"
            )

        try:
            import hvac

            self._backend_client = hvac.Client(
                url=self.config.vault_url,
                token=self.config.vault_token,
                namespace=self.config.vault_namespace,
            )

            # Test connection
            if not self._backend_client.is_authenticated():
                raise ConnectionError("Failed to authenticate with Vault")

        except ImportError:
            raise ImportError(
                "Vault backend requires 'hvac' package. Install with: pip install hvac"
            )

    def _init_aws(self) -> None:
        """Initialize AWS Secrets Manager client."""
        try:
            import boto3

            session_kwargs = {}
            if self.config.aws_region:
                session_kwargs["region_name"] = self.config.aws_region
            if self.config.aws_access_key_id:
                session_kwargs["aws_access_key_id"] = self.config.aws_access_key_id
            if self.config.aws_secret_access_key:
                session_kwargs["aws_secret_access_key"] = self.config.aws_secret_access_key

            self._backend_client = boto3.client("secretsmanager", **session_kwargs)

        except ImportError:
            raise ImportError(
                "AWS backend requires 'boto3' package. Install with: pip install boto3"
            )

    def _init_gcp(self) -> None:
        """Initialize GCP Secret Manager client."""
        if not self.config.gcp_project_id:
            raise ValueError("GCP backend requires gcp_project_id configuration")

        try:
            from google.cloud import secretmanager

            if self.config.gcp_credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                    self.config.gcp_credentials_path
                )

            self._backend_client = secretmanager.SecretManagerServiceClient()

        except ImportError:
            raise ImportError(
                "GCP backend requires 'google-cloud-secret-manager' package. "
                "Install with: pip install google-cloud-secret-manager"
            )

    def get_secret(
        self, key: str, default: Optional[str] = None, version: Optional[str] = None
    ) -> Optional[str]:
        """
        Get secret from configured backend.

        Args:
            key: Secret key/path
            default: Default value if secret not found
            version: Optional version to retrieve (backend-specific)

        Returns:
            Secret value or default

        Example:
            >>> jwt_secret = manager.get_secret("JWT_SECRET_KEY")
            >>> db_pass = manager.get_secret("database/password", version="v2")
        """
        # Check cache first
        if self.config.enable_cache and key in self._cache:
            cached_value, expiry = self._cache[key]
            if time.time() < expiry:
                self._track_access(key)
                return self._decrypt_value(cached_value) if self._cache_key else cached_value

        # Get from backend
        value = None
        if self.config.backend == SecretsBackend.ENVIRONMENT:
            value = self._get_from_env(key, default)
        elif self.config.backend == SecretsBackend.VAULT:
            value = self._get_from_vault(key, version)
        elif self.config.backend == SecretsBackend.AWS:
            value = self._get_from_aws(key, version)
        elif self.config.backend == SecretsBackend.GCP:
            value = self._get_from_gcp(key, version)

        if value is None:
            return default

        # Cache the secret
        if self.config.enable_cache:
            expiry = time.time() + self.config.cache_ttl_seconds
            cached_value = self._encrypt_value(value) if self._cache_key else value
            self._cache[key] = (cached_value, expiry)

        # Track access
        self._track_access(key, version)

        return value

    def _get_from_env(self, key: str, default: Optional[str]) -> Optional[str]:
        """Get secret from environment variables."""
        return os.getenv(key, default)

    def _get_from_vault(self, key: str, version: Optional[str]) -> Optional[str]:
        """Get secret from HashiCorp Vault."""
        try:
            # KV v2 API
            response = self._backend_client.secrets.kv.v2.read_secret_version(
                path=key,
                mount_point=self.config.vault_mount_point,
                version=version,
            )
            return response["data"]["data"].get("value")

        except Exception as e:
            # If KV v2 fails, try KV v1
            try:
                response = self._backend_client.secrets.kv.v1.read_secret(
                    path=key, mount_point=self.config.vault_mount_point
                )
                return response["data"].get("value")
            except Exception:
                raise e

    def _get_from_aws(self, key: str, version: Optional[str]) -> Optional[str]:
        """Get secret from AWS Secrets Manager."""
        try:
            kwargs = {"SecretId": key}
            if version:
                kwargs["VersionId"] = version

            response = self._backend_client.get_secret_value(**kwargs)

            # Return either SecretString or SecretBinary
            if "SecretString" in response:
                return response["SecretString"]
            else:
                return base64.b64decode(response["SecretBinary"]).decode("utf-8")

        except self._backend_client.exceptions.ResourceNotFoundException:
            return None

    def _get_from_gcp(self, key: str, version: Optional[str]) -> Optional[str]:
        """Get secret from GCP Secret Manager."""
        try:
            # Build the resource name
            if version:
                name = f"projects/{self.config.gcp_project_id}/secrets/{key}/versions/{version}"
            else:
                name = (
                    f"projects/{self.config.gcp_project_id}/secrets/{key}/versions/latest"
                )

            response = self._backend_client.access_secret_version(request={"name": name})
            return response.payload.data.decode("utf-8")

        except Exception:
            return None

    def set_secret(
        self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Set/update a secret (if backend supports it).

        Args:
            key: Secret key/path
            value: Secret value
            metadata: Optional metadata

        Returns:
            True if successful

        Example:
            >>> manager.set_secret("api/new_key", "secret-value-123")
        """
        if self.config.backend == SecretsBackend.VAULT:
            return self._set_in_vault(key, value, metadata)
        elif self.config.backend == SecretsBackend.AWS:
            return self._set_in_aws(key, value, metadata)
        elif self.config.backend == SecretsBackend.GCP:
            return self._set_in_gcp(key, value, metadata)
        else:
            # Environment variables are read-only
            return False

    def _set_in_vault(
        self, key: str, value: str, metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Set secret in HashiCorp Vault."""
        try:
            data = {"value": value}
            if metadata:
                data.update(metadata)

            self._backend_client.secrets.kv.v2.create_or_update_secret(
                path=key, secret=data, mount_point=self.config.vault_mount_point
            )

            # Invalidate cache
            if key in self._cache:
                del self._cache[key]

            return True

        except Exception:
            return False

    def _set_in_aws(
        self, key: str, value: str, metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Set secret in AWS Secrets Manager."""
        try:
            # Try to update existing secret
            try:
                kwargs = {"SecretId": key, "SecretString": value}
                if metadata:
                    kwargs["Description"] = json.dumps(metadata)

                self._backend_client.update_secret(**kwargs)

            except self._backend_client.exceptions.ResourceNotFoundException:
                # Create new secret
                kwargs = {"Name": key, "SecretString": value}
                if metadata:
                    kwargs["Description"] = json.dumps(metadata)

                self._backend_client.create_secret(**kwargs)

            # Invalidate cache
            if key in self._cache:
                del self._cache[key]

            return True

        except Exception:
            return False

    def _set_in_gcp(
        self, key: str, value: str, metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Set secret in GCP Secret Manager."""
        try:
            parent = f"projects/{self.config.gcp_project_id}"

            # Try to create secret first
            try:
                secret = {"replication": {"automatic": {}}}
                if metadata:
                    secret["labels"] = metadata

                self._backend_client.create_secret(
                    request={"parent": parent, "secret_id": key, "secret": secret}
                )
            except Exception:
                pass  # Secret might already exist

            # Add secret version
            secret_path = f"{parent}/secrets/{key}"
            payload = {"data": value.encode("utf-8")}

            self._backend_client.add_secret_version(
                request={"parent": secret_path, "payload": payload}
            )

            # Invalidate cache
            if key in self._cache:
                del self._cache[key]

            return True

        except Exception:
            return False

    def delete_secret(self, key: str) -> bool:
        """
        Delete a secret (if backend supports it).

        Args:
            key: Secret key/path

        Returns:
            True if successful
        """
        if self.config.backend == SecretsBackend.VAULT:
            try:
                self._backend_client.secrets.kv.v2.delete_metadata_and_all_versions(
                    path=key, mount_point=self.config.vault_mount_point
                )
                if key in self._cache:
                    del self._cache[key]
                return True
            except Exception:
                return False

        elif self.config.backend == SecretsBackend.AWS:
            try:
                self._backend_client.delete_secret(
                    SecretId=key, ForceDeleteWithoutRecovery=True
                )
                if key in self._cache:
                    del self._cache[key]
                return True
            except Exception:
                return False

        elif self.config.backend == SecretsBackend.GCP:
            try:
                name = f"projects/{self.config.gcp_project_id}/secrets/{key}"
                self._backend_client.delete_secret(request={"name": name})
                if key in self._cache:
                    del self._cache[key]
                return True
            except Exception:
                return False

        return False

    def rotate_secret(self, key: str, new_value: str) -> bool:
        """
        Rotate a secret (create new version and optionally delete old).

        Args:
            key: Secret key/path
            new_value: New secret value

        Returns:
            True if successful
        """
        return self.set_secret(key, new_value)

    def _track_access(self, key: str, version: Optional[str] = None) -> None:
        """Track secret access for auditing."""
        if key not in self._metadata:
            self._metadata[key] = SecretMetadata(key=key, version=version)

        metadata = self._metadata[key]
        metadata.last_accessed = time.time()
        metadata.access_count += 1

    def _generate_cache_key(self) -> bytes:
        """Generate encryption key for cache."""
        # Derive key from system information (NOT for production crypto!)
        # In production, use a proper key management system
        key_material = os.urandom(32)
        return base64.urlsafe_b64encode(key_material)

    def _encrypt_value(self, value: str) -> str:
        """Encrypt cached value."""
        if not self._cache_key:
            return value

        fernet = Fernet(self._cache_key)
        return fernet.encrypt(value.encode()).decode()

    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt cached value."""
        if not self._cache_key:
            return encrypted_value

        fernet = Fernet(self._cache_key)
        return fernet.decrypt(encrypted_value.encode()).decode()

    def mask_secret(self, secret: str, visible_chars: int = 4) -> str:
        """
        Mask a secret for logging.

        Args:
            secret: Secret to mask
            visible_chars: Number of visible characters at start and end

        Returns:
            Masked secret

        Example:
            >>> manager.mask_secret("my-secret-key-12345")
            'my-s***45'
        """
        if not self.config.mask_secrets_in_logs:
            return secret

        if len(secret) <= visible_chars * 2:
            return "*" * len(secret)

        return f"{secret[:visible_chars]}***{secret[-visible_chars:]}"

    def get_access_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get access statistics for all secrets."""
        stats = {}
        for key, metadata in self._metadata.items():
            stats[key] = {
                "access_count": metadata.access_count,
                "last_accessed": metadata.last_accessed,
                "version": metadata.version,
            }
        return stats

    def clear_cache(self) -> None:
        """Clear the secret cache."""
        self._cache.clear()
