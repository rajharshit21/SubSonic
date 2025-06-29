import sqlite3
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base



conn = sqlite3.connect("database/analytics.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS transformations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_file TEXT,
    filter TEXT,
    styled_text TEXT,
    output_file TEXT,
    timestamp TEXT
)
""")
conn.commit()

def log_transformation(original_file, filter, styled_text, output_file):
    timestamp = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO transformations (original_file, filter, styled_text, output_file, timestamp) VALUES (?, ?, ?, ?, ?)",
        (original_file, filter, styled_text, output_file, timestamp)
    )
    conn.commit()


Base = declarative_base()

class TransformationLog(Base):
    __tablename__ = 'transformation_logs'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    file_name = Column(String)
    filters_applied = Column(String)  # ✅ was 'filters_used'
    style_prompt = Column(String, nullable=True)  # ✅ newly added
    duration = Column(Float)
    user_id = Column(String, nullable=True)