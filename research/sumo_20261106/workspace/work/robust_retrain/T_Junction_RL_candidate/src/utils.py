import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ("W", "E", "S")


def load_yaml(name):
    path = Path(name)
    if not path.exists():
        path = ROOT / "configs" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binary(name):
    if not os.environ.get("SUMO_HOME"):
        try:
            import sumo
            os.environ["SUMO_HOME"] = str(Path(sumo.__file__).parent)
        except ImportError:
            pass
    result = shutil.which(name)
    if result:
        return result
    suffix = ".exe" if os.name == "nt" else ""
    candidates = [Path(sysconfig.get_path("scripts")) / (name + suffix),
                  Path.home() / ".local/bin" / name]
    if os.environ.get("SUMO_HOME"):
        candidates.insert(0, Path(os.environ["SUMO_HOME"]) / "bin" / (name + suffix))
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"{name} not found. Run: python -m pip install eclipse-sumo==1.28.0")


def version_info():
    import platform
    import torch
    return {"python": platform.python_version(), "sumo": subprocess.check_output(
        [binary("sumo"), "--version"], text=True).splitlines()[0],
        "torch": str(torch.__version__), "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
