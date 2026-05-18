# src/config.py
# Re-import the central config so all modules see the same settings.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config as _config

# Expose all attributes
for attr in dir(_config):
    if not attr.startswith("_"):
        globals()[attr] = getattr(_config, attr)