"""
Authentication utilities for Fortif.ai RAG API.
Implements API key-based authentication.
"""

import secrets
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from config import settings


# API Key header configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Verify API key from request header.

    This dependency can be used to protect endpoints:
        @app.post("/api/query", dependencies=[Depends(verify_api_key)])

    Args:
        api_key: API key from X-API-Key header

    Returns:
        Validated API key string

    Raises:
        HTTPException: If API key is missing or invalid
    """
    # If no API key is configured in settings, allow all requests (dev mode)
    if not settings.api_key:
        return "dev-mode-no-auth"

    # Check if API key was provided
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include 'X-API-Key' header in request.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Verify API key using constant-time comparison (prevents timing attacks)
    if not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return api_key


def generate_api_key(length: int = 32) -> str:
    """
    Generate a secure random API key.

    Utility function for administrators to generate new API keys.

    Args:
        length: Length of the API key (default 32 characters)

    Returns:
        Secure random API key string

    Example:
        >>> key = generate_api_key()
        >>> print(f"Your new API key: {key}")
    """
    return secrets.token_urlsafe(length)


# Optional: Role-based access control (future enhancement)
class APIKeyPermissions:
    """Define permission levels for different API keys."""

    READ_ONLY = "read"  # Can query, cannot ingest
    WRITE = "write"  # Can ingest data
    ADMIN = "admin"  # Full access

    @staticmethod
    def check_permission(api_key: str, required_permission: str) -> bool:
        """
        Check if API key has required permission level.

        Args:
            api_key: Validated API key
            required_permission: Required permission level

        Returns:
            True if permission granted

        Note:
            Current implementation grants all permissions.
            Extend this for production with key-to-permission mapping.
        """
        # TODO: Implement key-to-permission mapping
        # For now, all valid keys have all permissions
        return True
