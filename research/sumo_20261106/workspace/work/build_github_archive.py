"""Prepare a faithful, size-bounded research archive in an isolated worktree."""
import hashlib,json,os,re,shutil,sys,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs'
TARGET=ROOT/'work/github_publish/research/sumo_20261106'
TEMP=ROOT/'work/publish_archive_tmp'
CHUNK=24*1024*1024


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def scan(root):
    patterns=[re.compile(rb'ghp_[A-Za-z0-9]{30,}'),re.compile(rb'github_pat_[A-Za-z0-9_]{30,}'),
              re.compile(rb'AIza[0-9A-Za-z_-]{30,}'),re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
              re.compile(rb'xox[baprs]-[0-9A-Za-z-]{20,}')]
    bad=[];count=0
    suffixes={'.md','.txt','.py','.yaml','.yml','.json','.ipynb','.csv','.log','.ps1','.xml','.html'}
    def check(label,b):
        nonlocal count
        count+=1
        if any(p.search(b) for p in patterns):bad.append(label)
    for p in root.rglob('*'):
        if not p.is_file():continue
        if p.suffix.lower() in suffixes and p.stat().st_size<=4*1024*1024:
            check(str(p.relative_to(ROOT)),p.read_bytes())
        elif p.suffix.lower()=='.zip':
            try:
                with zipfile.ZipFile(p) as z:
                    for i in z.infolist():
                        if not i.is_dir() and i.file_size<=4*1024*1024 and Path(i.filename).suffix.lower() in suffixes:
                            check(f'{p.name}:{i.filename}',z.read(i))
            except zipfile.BadZipFile:bad.append(f'{p.name}:BAD_ZIP')
    print('SCANNED_TEXT_ITEMS',count,'FLAGGED_PATHS',bad,flush=True)
    if bad:raise RuntimeError('Potential credential pattern or invalid ZIP; inspect before publishing')


def copy_file(src,relative):
    dst=TARGET/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)


def write_chunks(file,logical,items):
    size=file.stat().st_size;h=digest(file);entry={'logical_path':logical,'size':size,'sha256':h,'parts':[]}
    base=TARGET/'artifacts'/f'{len(items)+1:03d}'
    base.mkdir(parents=True,exist_ok=True)
    with file.open('rb') as src:
        idx=1
        while True:
            b=src.read(CHUNK)
            if not b:break
            name=f'part{idx:03d}.bin';p=base/name;p.write_bytes(b)
            entry['parts'].append({'path':p.relative_to(TARGET).as_posix(),'size':len(b),'sha256':hashlib.sha256(b).hexdigest()})
            idx+=1
    items.append(entry)
    print('ARCHIVED',logical,'size_MiB',round(size/1048576,1),'parts',len(entry['parts']),flush=True)


def build():
    if TARGET.exists() and any(TARGET.iterdir()):raise RuntimeError('Destination already populated; inspect before continuing')
    TARGET.mkdir(parents=True,exist_ok=True);TEMP.mkdir(parents=True,exist_ok=True)
    scan(SOURCE)
    for p in sorted(SOURCE.glob('*')):
        if p.is_file() and p.suffix.lower() in ('.md','.ipynb','.png','.html','.py'):
            kind={'md':'reports','ipynb':'notebooks','png':'figures','html':'figures','py':'interfaces'}[p.suffix.lower()[1:]]
            copy_file(p,f'{kind}/{p.name}')
    for p in sorted((ROOT/'work').iterdir()):
        if p.is_file() and p.suffix.lower() in ('.py','.ps1','.txt','.json'):
            copy_file(p,f'workspace/work/{p.name}')
    project_roots=[ROOT/'work/robust_retrain'/n for n in ('T_Junction_RL','T_Junction_RL_candidate','T_Junction_RL_finetune','T_Junction_RL_reward_ablation')]
    project_roots += [ROOT/'work/colab_study/T_Junction_RL',ROOT/'work/fairness/T_Junction_RL']
    for project in project_roots:
        if not project.exists():continue
        for part in ('src','configs','maps','tests','scripts'):
            d=project/part
            if not d.exists():continue
            for p in d.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:
                    copy_file(p,Path('workspace')/p.relative_to(ROOT))
        for n in ('requirements.txt','pyproject.toml'):
            p=project/n
            if p.is_file():copy_file(p,Path('workspace')/p.relative_to(ROOT))
    for n in ('old_seed42.pt','robust_source.zip','finetune_source.zip','build_notebook.py','smoke.log','smoke2.log','resume.log','resume2.log','eval_smoke.log','tests.log'):
        p=ROOT/'work/robust_retrain'/n
        if p.is_file():copy_file(p,Path('workspace')/p.relative_to(ROOT))
    for folder in ('colab_analysis','robust_study','robust_study_v2','reassessment','transfer','result_audit','gui_test','reproduction'):
        p=ROOT/'work'/folder
        if p.exists():
            for f in p.iterdir():
                if f.is_file() and f.suffix.lower() in ('.py','.md','.txt','.json','.csv'):
                    copy_file(f,Path('workspace')/f.relative_to(ROOT))
    items=[]
    # Store every user-facing output file, including original research bundles.
    for p in sorted(SOURCE.iterdir()):
        if p.is_file():write_chunks(p,f'outputs/{p.name}',items)
    # Store every generated output directory, including all per-episode logs.
    for idx,p in enumerate(sorted(x for x in SOURCE.iterdir() if x.is_dir()),1):
        archive=TEMP/f'output_dir_{idx:03d}.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
            for f in sorted(p.rglob('*')):
                if f.is_file():z.write(f,f.relative_to(ROOT).as_posix())
        write_chunks(archive,f'archives/{p.name}.zip',items)
    (TARGET/'artifact_manifest.json').write_text(json.dumps({'chunk_bytes':CHUNK,'artifacts':items},indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('ARTIFACTS',len(items),'TOTAL_MIB',round(sum(x['size'] for x in items)/1048576,1),flush=True)


if __name__=='__main__':build()
