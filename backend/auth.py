import os
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

# Initialize Firebase Admin
# In production, use service account credentials. For local/CI, we use the project_id.
PROJECT_ID = "retinai-dashboard-102044"

try:
    firebase_admin.initialize_app()
except ValueError:
    # Already initialized
    pass

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class TokenData(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    uid: Optional[str] = None

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        # Verify the ID token sent from the frontend
        decoded_token = firebase_auth.verify_id_token(token)
        uid = decoded_token.get('uid')
        email = decoded_token.get('email')
        name = decoded_token.get('name', email.split('@')[0] if email else 'User')
        
        # Determine role from custom claims or email
        role = decoded_token.get('role')
        if not role:
            if email == "adityasa2838@gmail.com":
                role = "admin"
            else:
                role = "doctor"
                
        return TokenData(username=name, email=email, role=role, uid=uid)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

def check_role(allowed_roles: list):
    async def role_checker(current_user: TokenData = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this resource"
            )
        return current_user
    return role_checker
