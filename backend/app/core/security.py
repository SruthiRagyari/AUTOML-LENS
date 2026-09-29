"""Security utilities for file upload sanitization and path traversal prevention."""
import os
import re
import uuid
from pathlib import Path


ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and injection."""
    # Remove path separators
    filename = os.path.basename(filename)
    # Remove potentially dangerous characters
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    # Collapse whitespace
    filename = re.sub(r'\s+', '_', filename.strip())
    # Ensure it has an extension
    if not filename:
        filename = "uploaded_file"
    return filename


def validate_file_extension(filename: str) -> bool:
    """Check if file extension is allowed."""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def generate_safe_filename(original_filename: str) -> str:
    """Generate a unique, safe filename preserving the extension."""
    ext = Path(original_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".csv"
    unique_id = uuid.uuid4().hex[:12]
    safe_base = sanitize_filename(Path(original_filename).stem)[:50]
    return f"{safe_base}_{unique_id}{ext}"


def validate_path_safety(path: Path, allowed_root: Path) -> bool:
    """Ensure path doesn't escape the allowed directory."""
    try:
        resolved = path.resolve()
        allowed = allowed_root.resolve()
        return str(resolved).startswith(str(allowed))
    except (OSError, ValueError):
        return False
