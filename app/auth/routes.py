from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from app.models.user import UserCreate, User, Token
from app.auth.utils import authenticate_user, create_access_token, get_password_hash, get_current_user
from app.config import settings
from database.db import user_collection
from datetime import datetime
from bson import ObjectId

router = APIRouter()

# -------------------------------
# Register a new user
# -------------------------------
@router.post("/register", response_model=User)
async def register_user(user_data: UserCreate):
    """Register a new user."""
    # Check if email already exists
    if await user_collection.find_one({"email": user_data.email}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    user_in_db = {
        "email": user_data.email,
        "name": user_data.name,
        "hashed_password": hashed_password,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    result = await user_collection.insert_one(user_in_db)
    user_in_db["_id"] = result.inserted_id
    
    return User(
        id=str(user_in_db["_id"]),
        email=user_in_db["email"],
        name=user_in_db["name"],
        created_at=user_in_db["created_at"],
        updated_at=user_in_db["updated_at"]
    )


# -------------------------------
# Login and get access token
# -------------------------------
@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and generate token."""
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# -------------------------------
# Get current authenticated user
# -------------------------------
@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Return the current authenticated user."""
    return current_user
