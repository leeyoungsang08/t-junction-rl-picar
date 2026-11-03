#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
import importlib.util
from pathlib import Path
import subprocess
import sys
requirements = [x.strip() for x in Path('requirements.txt').read_text().splitlines() if x.strip() and not x.startswith('torch')]
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', *requirements])
if importlib.util.find_spec('torch') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', '--index-url', 'https://download.pytorch.org/whl/cpu'])
else:
    print('Keeping the installed PyTorch build.')
PY
python -m scripts.check_environment
