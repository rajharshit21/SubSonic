# api/analytics.py

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional, List
from database.session_logger import get_db

from database.models import TransformationLog

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/filters")
def get_filter_usage(db: Session = Depends(get_db)):
    """
    Returns frequency count of all filters used in transformations.
    """
    results = db.query(TransformationLog).all()
    counter = Counter()
    for row in results:
        if row.filters_applied:
            for f in row.filters_applied.split(','):
                counter[f.strip()] += 1
    return counter


@router.get("/daily")
def get_daily_counts(db: Session = Depends(get_db)):
    """
    Returns number of transformations per day (for line chart).
    """
    results = db.query(TransformationLog).all()
    daily = defaultdict(int)
    for row in results:
        day = row.timestamp.date().isoformat()
        daily[day] += 1
    return sorted(daily.items())  # [(date, count), ...]


@router.get("/sessions")
def get_sessions(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    filter_name: Optional[str] = Query(None, description="Filter name to search"),
    user_id: Optional[str] = Query(None, description="User ID (optional)"),
    db: Session = Depends(get_db)
):
    """
    Returns detailed session logs with optional filtering.
    """
    query = db.query(TransformationLog)

    if start_date:
        try:
            dt_start = datetime.fromisoformat(start_date)
            query = query.filter(TransformationLog.timestamp >= dt_start)
        except ValueError:
            return {"error": "Invalid start_date format. Use YYYY-MM-DD."}

    if end_date:
        try:
            dt_end = datetime.fromisoformat(end_date)
            query = query.filter(TransformationLog.timestamp <= dt_end)
        except ValueError:
            return {"error": "Invalid end_date format. Use YYYY-MM-DD."}

    if filter_name:
        query = query.filter(TransformationLog.filters_applied.contains(filter_name))

    if user_id:
        query = query.filter(TransformationLog.user_id == user_id)

    results = query.order_by(TransformationLog.timestamp.desc()).all()

    return [
        {
            "id": row.id,
            "file_name": row.file_name,
            "user_id": row.user_id,
            "filters_applied": row.filters_applied,
            "duration": row.duration,
            "style_prompt": row.style_prompt,
            "timestamp": row.timestamp.isoformat()
        }
        for row in results
    ]
