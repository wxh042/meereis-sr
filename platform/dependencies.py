from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

try:
    from .database import SessionLocal, User
    from .security import ALGORITHM, SECRET_KEY
except ImportError:
    from database import SessionLocal, User
    from security import ALGORITHM, SECRET_KEY


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token 无效或已过期",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")
        if not username or not role:
            raise credentials_exception
    except JWTError as exc:
        raise credentials_exception from exc

    with SessionLocal() as db:
        current_user = db.query(User).filter(User.username == username).first()
        if current_user is None:
            raise credentials_exception
        if current_user.role != role:
            raise credentials_exception
        db.expunge(current_user)
        return current_user


def verify_researcher_permission(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role not in {"admin", "researcher"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="权限不足，只有研究员及以上权限可调用GPU算力",
        )
    return current_user
