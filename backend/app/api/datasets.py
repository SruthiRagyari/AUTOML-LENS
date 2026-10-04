"""Dataset upload and profiling API routes."""
import json
import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd

from app.core.config import settings
from app.core.database import get_db, Dataset
from app.core.security import validate_file_extension, generate_safe_filename
from app.services.profiler import DatasetProfiler
from app.services.suitability import assess_suitability
from app.utils.file_utils import load_dataframe

logger = logging.getLogger(__name__)
router = APIRouter()


def _unusable_structure(df: pd.DataFrame) -> Optional[str]:
    """Return an error message when the parsed frame can never be used.

    Only whole-file structural problems are rejected here. Data-quality
    problems (missing values, a constant target, a tiny dataset) are reported
    by the suitability endpoint as warnings or blocking issues instead, so a
    usable file is never rejected over something the user can still act on.
    """
    rows, cols = df.shape
    if cols == 0:
        return "The file parsed successfully but contains no columns."
    if rows == 0:
        return ("The file has a header row but no data rows. Add at least "
                "a few rows of data and upload it again.")
    if cols < 2:
        return (f"The file has only {cols} column. AutoML needs at least one "
                "feature column plus a target column.")
    if bool(df.isnull().to_numpy().all()):
        return "Every cell in the file is empty."
    return None


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
        df = load_dataframe(file_path)
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to parse file: {str(e)}")

    # Structural validation. These files can never produce an experiment, so
    # they are rejected at the door with an explanatory message rather than
    # accepted and left to fail confusingly several screens later.
    problem = _unusable_structure(df)
    if problem:
        file_path.unlink(missing_ok=True)
        raise HTTPException(400, problem)

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

@router.get("")
async def list_datasets(db: Session = Depends(get_db)):
    """List all uploaded datasets with metadata."""
    datasets = db.query(Dataset).order_by(Dataset.id.desc()).all()
    return [
        {
            "id": ds.id,
            "filename": ds.filename,
            "original_filename": ds.original_filename,
            "rows": ds.rows,
            "columns": ds.columns,
            "file_size": ds.file_size,
            "uploaded_at": str(ds.uploaded_at),
            "has_profile": ds.profile_json is not None,
        }
        for ds in datasets
    ]


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


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Delete a dataset and its backing file if safe."""
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")

    from app.core.database import Experiment
    exps = db.query(Experiment).filter(Experiment.dataset_id == dataset_id).count()
    if exps > 0:
        raise HTTPException(400, f"Cannot delete dataset: referenced by {exps} experiment(s). Delete those experiments first.")

    if ds.file_path:
        try:
            p = Path(ds.file_path)
            if p.exists():
                p.unlink()
        except Exception as e:
            logger.warning(f"Could not delete dataset file {ds.file_path}: {e}")

    db.delete(ds)
    db.commit()
    return {"message": f"Dataset {dataset_id} deleted successfully", "id": dataset_id}


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
        df = load_dataframe(ds.file_path)
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


@router.get("/{dataset_id}/suitability")
async def dataset_suitability(dataset_id: int, target_column: str = None,
                              problem_type: str = None,
                              db: Session = Depends(get_db)):
    """Can this dataset drive an AutoML run, and is the target usable?

    Everything is measured from the stored file. ``target_column`` is optional:
    when omitted the profiler's suggestion is used, and when that is also
    absent the response explains that a target still has to be chosen.

    Returns blocking issues (cannot train meaningfully) separately from
    warnings (trainable, but worth knowing). No value is invented: unknown
    fields are ``null`` and explained in ``task_reason``.
    """
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    try:
        df = load_dataframe(ds.file_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to read dataset: {str(e)}")

    profiler = DatasetProfiler()
    profile = profiler.profile(df)
    suggested_target, suggested_type = profiler.suggest_target(profile)

    chosen = target_column or suggested_target
    chosen_type = problem_type if problem_type in (
        "classification", "regression") else None

    result = assess_suitability(
        df, profile["column_profiles"], chosen, chosen_type)

    return {
        "dataset_id": dataset_id,
        "original_filename": ds.original_filename,
        "filename": ds.filename,
        "suggested_target": suggested_target,
        "suggested_problem_type": suggested_type,
        "target_column": chosen,
        "problem_type_requested": problem_type,
        # Profile-level quality signals the UI also renders.
        "warnings_from_profile": profile.get("warnings", []),
        "column_type_summary": profile.get("column_type_summary", {}),
        "duplicate_rows": profile.get("duplicate_rows"),
        "duplicate_percentage": profile.get("duplicate_percentage"),
        **result,
    }


@router.get("/{dataset_id}/columns")
async def get_columns(dataset_id: int, db: Session = Depends(get_db)):
    """Get column names for a dataset."""
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    try:
        df = load_dataframe(ds.file_path, nrows=0)
        return {"columns": list(df.columns)}
    except Exception:
        return {"columns": []}
