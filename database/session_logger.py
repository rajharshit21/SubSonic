from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base, TransformationLog

DATABASE_URL = "sqlite:///./database/analytics.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables if not exists
Base.metadata.create_all(bind=engine)

def log_transformation(file_name: str, filters_used: list, duration: float, user_id: str = None):
    db = SessionLocal()
    try:
        log = TransformationLog(
            file_name=file_name,
            filters_used=",".join(filters_used),
            duration=duration,
            user_id=user_id
        )
        db.add(log)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to log transformation: {e}")
    finally:
        db.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
