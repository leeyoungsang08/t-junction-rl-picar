import hashlib
import json
from pathlib import Path
import time
import zipfile

work = Path(__file__).resolve().parent
root = work / 'T_Junction_RL'
out = work.parent / 'outputs' / 'T_Junction_RL_Project_Validated_Final.zip'
original = json.loads((work / 'source_zip_sha256.json').read_text())
(root / 'ORIGINAL_SHA256.json').write_text(json.dumps(original, indent=2, sort_keys=True)+'\n', encoding='utf-8')
manifest_name = 'RUN_MANIFEST.json'
pending = {
    manifest_name, 'docs/experiment_results.md', 'docs/failure_analysis.md', 'VERIFICATION_REPORT.md',
    'outputs/v3_research/verification/evidence_audit.json',
}

def paths_to_package():
    return {p.relative_to(root).as_posix(): p for p in sorted(root.rglob('*')) if p.is_file()
            and not {'__pycache__', '.pytest_cache', '.ipynb_checkpoints'}.intersection(p.relative_to(root).parts)
            and not p.name.endswith('.tmp')}

relative = paths_to_package()
assert set(original) <= set(relative), set(original) - set(relative)
preserved, modified = [], {}
for name, old in original.items():
    new = hashlib.sha256(relative[name].read_bytes()).hexdigest()
    if new == old:
        preserved.append(name)
    else:
        assert not name.startswith('outputs/'), name
        modified[name] = {'input': old, 'delivered': new}

required = {
    'README.md', 'CHANGELOG_V3_FINAL.md', 'VERIFICATION_REPORT.md', 'RUN_MANIFEST.json', 'ORIGINAL_SHA256.json',
    'docs/technical_audit.md', 'docs/literature_review.md', 'docs/paper_review_matrix.csv', 'docs/state_action_reward.md',
    'docs/experiment_design.md', 'docs/evaluation_protocol.md', 'docs/experiment_results.md', 'docs/failure_analysis.md',
    'docs/future_deployment.md', 'notebooks/07_paper_v3_experiments.ipynb', 'configs/scenarios.yaml',
    'maps/t_junction.net.xml', 'maps/t_junction.sumocfg', 'scripts/research_matrix.py', 'scripts/research_evaluation.py',
    'scripts/research_report.py', 'scripts/audit_research.py', 'src/environment.py', 'tests/test_research_regressions.py',
}
temporary = work / 'T_Junction_RL_Project_Validated_Final.zip.tmp'
manifest_path = root / manifest_name
with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as z:
    print(f'Archiving {len(relative):,} project paths before the final audit documents.', flush=True)
    for index, (name, p) in enumerate(relative.items(), start=1):
        if name not in pending:
            z.writestr(f'{root.name}/{name}', p.read_bytes())
        if index % 5000 == 0:
            print(f'Archived {index:,}/{len(relative):,} paths', flush=True)
    deadline = time.monotonic() + 600
    while not manifest_path.exists() and time.monotonic() < deadline:
        time.sleep(2)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for name in pending:
        assert (root / name).is_file(), name
        relative[name] = root / name
    required.update(manifest['required_evidence'])
    assert required <= relative.keys(), required - relative.keys()
    manifest['files'] = {'preserved': preserved, 'modified': modified, 'added': sorted(set(relative)-set(original))}
    source = Path('C:/Users/user/Downloads/T_Junction_RL_Project_Paper_V3.zip')
    manifest['source_archive'] = {'name': source.name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    manifest['archive_checks'] = ['CRC of every member', 'all 3607 original paths present', 'historical outputs SHA256',
                                  'required scripts, docs, notebook and executed experiment evidence']
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    for name in sorted(pending):
        z.writestr(f'{root.name}/{name}', relative[name].read_bytes())
print('Reopening archive to verify CRCs, historical SHA256 and required paths.', flush=True)
with zipfile.ZipFile(temporary) as z:
    assert z.testzip() is None
    assert set(z.namelist()) == {f'{root.name}/{name}' for name in relative}
    for name, h in original.items():
        if name.startswith('outputs/'):
            assert hashlib.sha256(z.read(f'{root.name}/{name}')).hexdigest() == h, name
    cfg = z.read(f'{root.name}/maps/t_junction.sumocfg').decode()
    assert 't_junction.net.xml' in cfg
    nb = json.loads(z.read(f'{root.name}/notebooks/07_paper_v3_experiments.ipynb'))
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), cell['id'], 'exec')
temporary.replace(out)
print(f'{out}\n{out.stat().st_size:,} bytes; {len(relative)} files; '
      f'{len(preserved)} preserved, {len(modified)} modified, {len(relative)-len(original)} added; CRC and historical SHA256 verified')
