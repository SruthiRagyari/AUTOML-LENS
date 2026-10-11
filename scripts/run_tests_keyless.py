"""Run the test suite with LLM API keys forced empty.

The suite is designed to run keyless: provider-specific integration tests
(e.g. tests/test_real_gemini.py) carry a module-level ``skipif`` on the
corresponding settings value and skip - never fall back - when the key is
missing.

An empty-string assignment cannot be expressed through the Windows shell
(``$env:X=''`` removes the variable instead of setting it empty), and a
``backend/.env`` file would otherwise supply a configured key via
``load_dotenv``. Setting ``os.environ[key] = ""`` before pytest imports the
application guarantees the keyless path regardless of the surrounding
machine or .env configuration.

Usage (from the repository root):
    python scripts/run_tests_keyless.py tests -q
"""
import os
import sys

os.environ["GEMINI_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

import pytest  # noqa: E402  (must follow the environment pinning above)

if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv[1:] if len(sys.argv) > 1 else ["tests", "-q"]))
