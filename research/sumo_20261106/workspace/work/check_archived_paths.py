import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import time
from zipfile import ZipFile

output = Path(__file__).resolve().parents[1] / 'outputs/T_Junction_RL_Project_Validated_Final.zip'
deadline = time.monotonic() + 900
while not output.is_file() and time.monotonic() < deadline:
    time.sleep(2)
root = 'T_Junction_RL/'
with ZipFile(output) as z:
    names = set(z.namelist())
    def content(path):
        return z.read(root + path)

    manifest = json.loads(content('RUN_MANIFEST.json'))
    assert len(names) == len(manifest['files']['preserved']) + len(manifest['files']['modified']) + len(manifest['files']['added'])
    for area in ('smoke', 'quick', 'full'):
        for name in ['validation_eval', *(['final_eval'] if area == 'full' else [])]:
            evaluation = f'outputs/v3_research/{area}/{name}/'
            plan = json.loads(content(evaluation + 'evaluation_plan.json'))
            rows = list(csv.DictReader(StringIO(content(evaluation + 'seed_level_results.csv').decode())))
            routes = json.loads(content(evaluation + 'demand_manifest.json'))
            assert len(rows) == len(plan['controllers']) * len(plan['scenarios']) * plan['seeds']
            assert len(routes) == len(plan['scenarios']) * plan['seeds']
            for row in rows:
                for trace in ('metrics.json', 'tripinfo.xml', 'timeseries.csv', 'decisions.csv'):
                    assert root + evaluation + row['episode_dir'] + '/' + trace in names
            assert hashlib.sha256(content('maps/t_junction.net.xml')).hexdigest() == plan['net_sha256']
            for label, expected in plan['model_sha256'].items():
                path = f'outputs/v3_research/{area}/{label}/checkpoints/best_model.pt'
                assert hashlib.sha256(content(path)).hexdigest() == expected, path
            for route in routes:
                assert hashlib.sha256(content(evaluation + route['route'])).hexdigest() == route['sha256']
            print(f'{area}/{name}: checked {len(rows)} episode paths, {len(routes)} route hashes, and {len(plan["model_sha256"])} model hashes')
    for name in names:
        if name.startswith(tuple(root + folder + '/' for folder in ('src','scripts','maps','tests'))) and name.endswith('.py'):
            compile(z.read(name), name, 'exec')
    assert manifest['tests']['passed'] == 48 and manifest['final_evaluation']['episodes'] == 990
    print('All archived Python entry points compile, and manifest counts match.')
with output.open('rb') as stream:
    print('ZIP SHA256', hashlib.file_digest(stream, 'sha256').hexdigest())
