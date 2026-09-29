"""Dataset upload and profiling API routes."""
import json
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd

from app.core.config import settings
from app.core.database import get_db, Dataset
from app.core.security import validate_file_extension, generate_safe_filename
from app.services.profiler import DatasetProfiler

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a CSV or XLSX dataset."""
    if not file.filename:
        raise HTTPException(400, "No file provided")
    if not validate_file_extension(file.filename):
        raise HTTPException(400, "Only .csv and .xlsx files are supported")

    # Read file content
    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 100MB)")

    # Save file
    safe_name = generate_safe_filename(file.filename)
    file_path = settings.datasets_path / safe_name
    with open(file_path, "wb") as f:
        f.write(content)

    # Parse dataset
    try:
        ext = Path(file.filename).suffix.lower()
        if ext == ".xlsx" or ext == ".xls":
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to parse file: {str(e)}")

    # Create DB record
    dataset = Dataset(
        filename=safe_name,
        original_filename=file.filename,
        file_path=str(file_path),
        file_size=len(content),
        rows=len(df),
        columns=len(df.columns),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return {
        "id": dataset.id,
        "filename": safe_name,
        "original_filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "file_size": len(content),
        "uploaded_at": str(dataset.uploaded_at),
        "column_names": list(df.columns),
    }


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Get dataset metadata."""
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    return {
        "id": ds.id,
        "filename": ds.filename,
        "original_filename": ds.original_filename,
        "rows": ds.rows,
        "columns": ds.columns,
        "file_size": ds.file_size,
        "uploaded_at": str(ds.uploaded_at),
        "has_profile": ds.profile_json is not None,
    }


@router.get("/{dataset_id}/profile")
async def profile_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Generate or retrieve dataset profile."""
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")

    # Return cached profile if available
    if ds.profile_json:
        return json.loads(ds.profile_json)

    # Generate profile
    try:
        ext = Path(ds.file_path).suffix.lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(ds.file_path)
        else:
            df = pd.read_csv(ds.file_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to read dataset: {str(e)}")

    profiler = DatasetProfiler()
    profile = profiler.profile(df)

    # Suggest target
    target, ptype = profiler.suggest_target(profile)
    profile["dataset_id"] = dataset_id
    profile["suggested_target"] = target
    profile["suggested_problem_type"] = ptype

    # Cache profile
    ds.profile_json = json.dumps(profile, default=str)
    db.commit()

    return profile


@router.get("/{dataset_id}/columns")
async def get_columns(dataset_id: int, db: Session = Depends(get_db)):
    """Get column names for a dataset."""
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    try:
        df = pd.read_csv(ds.file_path, nrows=0)
        return {"columns": list(df.columns)}
    except Exception:
        return {"columns": []}
