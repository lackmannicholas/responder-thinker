"""
Root conftest — set required env vars before any backend modules are imported.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")
