"""Location of the assistant's local application database."""

import os
from pathlib import Path

APP_DB_PATH = Path(os.environ.get("ASSISTANT_DB_PATH", "data/assistant.db"))
