"""File utility functions."""
import os
from pathlib import Path


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if needed."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_file_size(path: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def safe_path_join(base: str, *parts: str) -> str:
    """Safely join path components preventing traversal."""
    result = os.path.normpath(os.path.join(base, *parts))
    if not result.startswith(os.path.normpath(base)):
        raise ValueError("Path traversal detected")
    return result
