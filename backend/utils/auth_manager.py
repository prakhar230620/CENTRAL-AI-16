import jwt
import datetime
from passlib.context import CryptContext
from fastapi import HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from backend.models.database_models import User
import logging
import re

# Constants
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_password_hash(password):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: datetime.timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        logger.warning(f"Failed login attempt for username: {username}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    if not verify_password(password, user.password_hash):
        logger.warning(f"Failed login attempt for username: {username}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    return user


def register_user(db: Session, username: str, email: str, password: str):
    if not validate_username(username):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid username")
    if not validate_email(email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid email")
    if not validate_password(password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid password")

    hashed_password = get_password_hash(password)
    db_user = User(username=username, email=email, password_hash=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    logger.info(f"User registered: {username}")
    return db_user


def get_current_user(db: Session, token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def update_user_profile(db: Session, user: User, **kwargs):
    for key, value in kwargs.items():
        if key == "email" and not validate_email(value):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid email")
        setattr(user, key, value)
    db.commit()
    db.refresh(user)
    logger.info(f"User profile updated: {user.username}")
    return user


def delete_user(db: Session, user: User):
    db.delete(user)
    db.commit()
    logger.info(f"User deleted: {user.username}")


def validate_username(username: str):
    if len(username) < 3 or len(username) > 50:
        return False
    return True


def validate_email(email: str):
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email) is not None


def validate_password(password: str):
    if len(password) < 8:
        return False
    return True