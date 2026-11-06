import json,re,zipfile
from pathlib import Path
from build_github_archive import TARGET,write_chunks

sources=[(Path('C:/Users/user/Downloads/T_Junction_RL_Project_Paper_V3.zip'),'inputs/T_Junction_RL_Project_Paper_V3.zip'),
         (Path('C:/Users/user/.codex/attachments/4f41e312-4df4-4270-8063-7cbaa3e8efd5/붙여넣은 텍스트.txt'),'inputs/USER_MASTER_PROMPT_KO.txt')]
patterns=[re.compile(rb'ghp_[A-Za-z0-9]{30,}'),re.compile(rb'github_pat_[A-Za-z0-9_]{30,}'),
          re.compile(rb'AIza[0-9A-Za-z_-]{30,}'),re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
          re.compile(rb'xox[baprs]-[0-9A-Za-z-]{20,}')]
exts={'.md','.txt','.py','.yaml','.yml','.json','.ipynb','.csv','.log','.ps1','.xml','.html'}
for p,_ in sources:
    if not p.is_file():raise FileNotFoundError(p)
    if p.suffix.lower()=='.zip':
        with zipfile.ZipFile(p) as z:
            for i in z.infolist():
                if not i.is_dir() and i.file_size<=4*1024*1024 and Path(i.filename).suffix.lower() in exts:
                    if any(q.search(z.read(i)) for q in patterns):raise ValueError(f'Potential credential: {i.filename}')
    elif any(q.search(p.read_bytes()) for q in patterns):raise ValueError(f'Potential credential in {p.name}')
manifest_path=TARGET/'artifact_manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf8'))
existing={x['logical_path'] for x in manifest['artifacts']}
for path,logical in sources:
    if logical in existing:raise RuntimeError(f'Already archived: {logical}')
    write_chunks(path,logical,manifest['artifacts'])
manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print('ORIGINAL_INPUTS_ADDED',len(sources),flush=True)
