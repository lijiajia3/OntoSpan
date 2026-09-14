"""Optional developer CPU tests. 02 never invokes these."""
import subprocess
import sys
from pathlib import Path
if __name__=='__main__':
    from windows_launcher import ensure_runtime
    ensure_runtime()
    raise SystemExit(subprocess.call([sys.executable,'-m','unittest','discover','-p','test_*.py','-v'],cwd=Path(__file__).resolve().parent))
