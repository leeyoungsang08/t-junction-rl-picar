import argparse
from pathlib import Path
import zipfile
from src.utils import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='../T_Junction_RL_results.zip')
    a = p.parse_args()
    destination = Path(a.output).resolve()
    development_outputs = {'smoke', 'verified_quick', 'pipeline_check', 'initial_sumo_check',
                           'playground_verification', 'encoding_check', 'map_preview'}
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(ROOT.rglob('*')):
            if not path.is_file() or path.resolve() == destination:
                continue
            if any(x in path.parts for x in ['__pycache__', '.pytest_cache', '.venv', '.git']):
                continue
            relative = path.relative_to(ROOT)
            if len(relative.parts) > 1 and relative.parts[0] == 'outputs' and relative.parts[1] in development_outputs:
                continue
            if path.suffix in ['.tmp', '.pyc', '.zip']:
                continue
            archive.write(path, Path("T_Junction_RL") / path.relative_to(ROOT))
    print(destination)


if __name__ == '__main__':
    main()
