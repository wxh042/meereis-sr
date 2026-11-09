import os
import logging
from datetime import datetime

import pymysql  # noqa: F401 - required by the mysql+pymysql SQLAlchemy dialect.
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import Session, declarative_base, sessionmaker


SQLALCHEMY_DATABASE_URL = os.getenv(
    "SQLALCHEMY_DATABASE_URL",
    os.getenv(
        "DATABASE_URL",
        "sqlite:///./meereis_demo.db",
    ),
)

engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True,
    **({"connect_args": {"check_same_thread": False}} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
logger = logging.getLogger(__name__)

TASK_STATUS_QUEUED = "QUEUED"
TASK_STATUS_STARTED = "STARTED"
TASK_STATUS_SUCCESS = "SUCCESS"
TASK_STATUS_FAILURE = "FAILURE"
VALID_TASK_STATUSES = {
    TASK_STATUS_QUEUED,
    TASK_STATUS_STARTED,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE,
}


class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    operator_name = Column(String(100), nullable=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    original_filename = Column(String(255), nullable=True)
    lr_path = Column(String(500), nullable=True)
    sr_path = Column(String(500), nullable=True)
    status = Column(String(50), nullable=False, default=TASK_STATUS_SUCCESS)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    # RBAC roles: admin, researcher, guest.
    role = Column(String(32), nullable=False, default="guest", comment="admin, researcher, guest")


class TaskHistory(Base):
    __tablename__ = "task_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    operator_name = Column(String(100), nullable=True)
    task_id = Column(String(100), unique=True, nullable=False, index=True)
    task_type = Column(String(20), nullable=False, default="SINGLE", comment="SINGLE or BATCH")
    model_version = Column(String(20), nullable=False, default="25km", comment="25km or 10km")
    status = Column(String(50), nullable=False, default=TASK_STATUS_QUEUED)
    result_path = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


def get_db() -> Session:
    """Create a short-lived database session for request/task scoped work."""
    return SessionLocal()


def init_db() -> None:
    """Initialize MySQL tables through SQLAlchemy metadata."""
    Base.metadata.create_all(bind=engine)
    seed_default_users()


def seed_default_users() -> None:
    """Create default JWT demo users when they do not already exist."""
    try:
        from .security import get_password_hash
    except ImportError:
        from security import get_password_hash

    demo_password = os.getenv("DEMO_PASSWORD")
    default_users = ([{"username": "researcher", "password": demo_password, "role": "researcher"}]
                     if demo_password else [])

    with SessionLocal() as db:
        for user_data in default_users:
            existing_user = db.execute(select(User).where(User.username == user_data["username"])).scalar_one_or_none()
            if existing_user is not None:
                continue
            db.add(
                User(
                    username=user_data["username"],
                    hashed_password=get_password_hash(user_data["password"]),
                    role=user_data["role"],
                )
            )
        db.commit()


def add_record(operator, filename, lr_path, sr_path, status=TASK_STATUS_SUCCESS) -> None:
    """Persist one processing history record."""
    with SessionLocal() as db:
        db.add(
            History(
                operator_name=operator,
                timestamp=datetime.now(),
                original_filename=filename,
                lr_path=lr_path,
                sr_path=sr_path,
                status=status,
            )
        )
        db.commit()


def update_task_history_status(
    task_id: str | None = None,
    status: str = TASK_STATUS_QUEUED,
    operator_name: str | None = None,
    result_path: str | None = None,
    error_message: str | None = None,
    model_version: str | None = None,
) -> bool:
    """Update a task history row by Celery task id."""
    if status not in VALID_TASK_STATUSES:
        raise ValueError(f"Invalid task status: {status}. Expected one of {sorted(VALID_TASK_STATUSES)}.")

    with SessionLocal() as db:
        task_history = db.execute(select(TaskHistory).where(TaskHistory.task_id == task_id)).scalar_one_or_none()
        if task_history is None:
            logger.error("TaskHistory update failed: task_id=%s was not found.", task_id)
            return False

        task_history.status = status
        task_history.timestamp = datetime.now()
        task_history.updated_at = datetime.now()
        if operator_name is not None:
            task_history.operator_name = operator_name
        if model_version is not None:
            task_history.model_version = model_version
        if result_path is not None:
            task_history.result_path = result_path
        if error_message is not None:
            task_history.error_message = error_message
        db.commit()
        return True


def update_task_history(
    task_id: str | None = None,
    status: str = TASK_STATUS_QUEUED,
    operator_name: str | None = None,
    result_path: str | None = None,
    error_message: str | None = None,
    model_version: str | None = None,
) -> bool:
    """Compatibility wrapper for worker task history updates."""
    return update_task_history_status(
        task_id=task_id,
        status=status,
        operator_name=operator_name,
        result_path=result_path,
        error_message=error_message,
        model_version=model_version,
    )


def get_all_records():
    """Read all history records in the tuple order expected by the Streamlit UI."""
    with SessionLocal() as db:
        rows = db.execute(select(History).order_by(History.timestamp.desc())).scalars().all()
        return [
            (
                row.id,
                row.operator_name,
                row.timestamp,
                row.original_filename,
                row.lr_path,
                row.sr_path,
                row.status,
            )
            for row in rows
        ]


if __name__ == "__main__":
    init_db()
    print("MySQL database initialized successfully.")
