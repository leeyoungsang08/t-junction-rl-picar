import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT, binary, version_info


def main():
    import sumo
    import traci
    import sumolib
    os.environ.setdefault("SUMO_HOME", str(Path(sumo.__file__).parent))
    os.environ["PATH"] = str(Path(os.environ["SUMO_HOME"]) / "bin") + os.pathsep + os.environ["PATH"]
    print(json.dumps({**version_info(), "SUMO_HOME": os.environ["SUMO_HOME"], "project_path": str(ROOT),
          "netconvert": subprocess.check_output([binary("netconvert"), "--version"], text=True).splitlines()[0]}, indent=2))


if __name__ == "__main__":
    main()
