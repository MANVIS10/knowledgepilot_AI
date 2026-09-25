import os
import sys
from pathlib import Path

# Make `src` importable and keep DeepEval from sending usage data
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
