"""Verify and restore split research artifacts without overwriting other files."""
import argparse,hashlib,json,os,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def safe_path(root,name):
    root=root.resolve();path=(root/name).resolve()
    if path!=root and root not in path.parents:raise ValueError(f'Unsafe path: {name}')
    return path


def extract_archive(target,dest):
    with zipfile.ZipFile(target) as z:
        for member in z.infolist():
            output=safe_path(dest,member.filename)
            if member.is_dir():output.mkdir(parents=True,exist_ok=True);continue
            output.parent.mkdir(parents=True,exist_ok=True)
            if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
            with z.open(member) as source,output.open('xb') as target_file:
                for block in iter(lambda:source.read(1024*1024),b''):target_file.write(block)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--destination',default='restored_research',help='directory for outputs and archives')
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--extract',action='store_true',help='also extract directory archives to destination')
    args=p.parse_args();dest=Path(args.destination).resolve()
    manifest=json.loads((ROOT/'artifact_manifest.json').read_text(encoding='utf8'))
    for item in manifest['artifacts']:
        target=safe_path(dest,item['logical_path']);combined=hashlib.sha256();total=0
        if not args.verify_only:
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if target.stat().st_size!=item['size'] or sha(target)!=item['sha256']:
                    raise FileExistsError(f'Existing file differs: {target}')
                print('EXISTS',target)
                if args.extract and item['logical_path'].startswith('archives/'):
                    extract_archive(target,dest)
                continue
            temp=target.with_name(target.name+'.restoring')
            if temp.exists():raise FileExistsError(f'Partial output exists: {temp}')
            stream=temp.open('xb')
        else:stream=None
        try:
            for part in item['parts']:
                path=safe_path(ROOT,part['path']);ph=hashlib.sha256();n=0
                with path.open('rb') as source:
                    for block in iter(lambda:source.read(1024*1024),b''):
                        ph.update(block);combined.update(block);n+=len(block);total+=len(block)
                        if stream:stream.write(block)
                if n!=part['size'] or ph.hexdigest()!=part['sha256']:raise ValueError(f'Corrupt part: {path}')
            if total!=item['size'] or combined.hexdigest()!=item['sha256']:raise ValueError(f'Corrupt artifact: {item["logical_path"]}')
            if stream:stream.close();stream=None;os.replace(temp,target)
        except BaseException:
            if stream:stream.close();stream=None
            if not args.verify_only:temp.unlink(missing_ok=True)
            raise
        finally:
            if stream:stream.close()
        print('VERIFIED',item['logical_path'])
        if args.extract and not args.verify_only and item['logical_path'].startswith('archives/'):
            extract_archive(target,dest)
    print('ALL ARTIFACTS VERIFIED')


if __name__=='__main__':main()
