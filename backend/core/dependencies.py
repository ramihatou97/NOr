# backend/core/dependencies.py
"""
Authentication and dependency injection utilities
"""

from typing import Optional
from pydantic import BaseModel
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)

class CurrentUser(BaseModel):
    """Current user model"""
    user_id: str
    username: str
    email: Optional[str] = None
    is_active: bool = True
    is_admin: bool = False
    specialty: Optional[str] = None

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> CurrentUser:
    """
    Dependency to get current authenticated user
    For now, returns a mock user for development
    """

    # In development mode, return a mock user
    # In production, this would validate the JWT token and return real user data
    if not credentials:
        # For development - allow unauthenticated access with default user
        logger.warning("No authentication credentials provided, using default user")
        return CurrentUser(
            user_id="dev_user_123",
            username="dev_user",
            email="dev@example.com",
            is_active=True,
            is_admin=True,
            specialty="general_medicine"
        )

    # TODO: Implement actual JWT token validation
    # For now, accept any token and return mock user
    token = credentials.credentials

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Mock user based on token (in real implementation, decode JWT)
    return CurrentUser(
        user_id="user_" + token[:8],
        username="authenticated_user",
        email="user@example.com",
        is_active=True,
        is_admin=False,
        specialty="neurosurgery"
    )

# Additional dependency functions can be added here
async def get_admin_user(
    current_user: CurrentUser = Depends(get_current_user)
) -> CurrentUser:
    """Dependency that requires admin privileges"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user