import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from app.core.config import settings

db_url = settings.database_url

try:
    # Quick probe of PostgreSQL connectivity
    probe_engine = create_engine(db_url, connect_args={"connect_timeout": 2})
    with probe_engine.connect() as conn:
        pass
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        echo=settings.debug,
    )
    print("[+] Database: Connected to PostgreSQL")
except Exception as e:
    sqlite_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../sahachaara.db"))
    db_url = f"sqlite:///{sqlite_path}"
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
        echo=settings.debug,
    )
    print(f"[!] Database: PostgreSQL connection refused. Falling back to SQLite: {db_url}")

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — yields a DB session and ensures cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
