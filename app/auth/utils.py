from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.config import settings
from app.models.user import TokenData, User, UserInDB
from database.db import user_collection

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


# -------------------------------
# Password verification and hashing
# -------------------------------
def verify_password(plain_password, hashed_password):
    """Verify password using bcrypt."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    """Hash password using bcrypt."""
    return pwd_context.hash(password)


# -------------------------------
# Fetch user by email
# -------------------------------
async def get_user(email: str):
    """Retrieve user from database."""
    user = await user_collection.find_one({"email": email})
    if user:
        return UserInDB(**user)
    return None


# -------------------------------
# Authenticate user
# -------------------------------
async def authenticate_user(email: str, password: str):
    """Authenticate user with email and password."""
    user = await get_user(email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


# -------------------------------
# Create JWT access token
# -------------------------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Generate JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


# -------------------------------
# Get current authenticated user
# -------------------------------
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Retrieve authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    user = await get_user(email=token_data.email)
    if user is None:
        raise credentials_exception
    return User(
        id=str(user.id),
        email=user.email,
        name=user.name,
        created_at=user.created_at,
        updated_at=user.updated_at
    )
