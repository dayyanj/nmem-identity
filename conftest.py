"""Put `src/` on the path so `import nmem_identity` works without an editable install.

Only the pytest unit suite (tests/test_*.py) is collected by default — the DB-backed smoke scripts
(tests/smoke_*.py) are run manually against a live/disposable DB, not collected here (no test_* funcs).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
