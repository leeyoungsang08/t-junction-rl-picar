import hashlib,json,subprocess,sys,zipfile
from pathlib import Path


SCRIPT=Path(__file__).resolve().parents[1]/'restore_artifacts.py'


def test_restore_verifies_parts_and_refuses_overwrite(tmp_path):
    root=tmp_path/'archive';root.mkdir();script=root/'restore_artifacts.py';script.write_bytes(SCRIPT.read_bytes())
    data=b'first part and second part';pieces=(data[:10],data[10:]);parts=[]
    for i,chunk in enumerate(pieces):
        p=root/f'part{i}.bin';p.write_bytes(chunk)
        parts.append({'path':p.name,'size':len(chunk),'sha256':hashlib.sha256(chunk).hexdigest()})
    record={'logical_path':'outputs/file.bin','size':len(data),'sha256':hashlib.sha256(data).hexdigest(),'parts':parts}
    (root/'artifact_manifest.json').write_text(json.dumps({'artifacts':[record]}),encoding='utf8')
    dest=tmp_path/'restored'
    subprocess.run([sys.executable,str(script),'--verify-only'],check=True,capture_output=True)
    subprocess.run([sys.executable,str(script),'--destination',str(dest)],check=True,capture_output=True)
    output=dest/'outputs/file.bin';assert output.read_bytes()==data
    output.write_bytes(b'user data')
    result=subprocess.run([sys.executable,str(script),'--destination',str(dest)],capture_output=True)
    assert result.returncode!=0 and output.read_bytes()==b'user data'
    (root/'part0.bin').write_bytes(b'corrupted!')
    result=subprocess.run([sys.executable,str(script),'--verify-only'],capture_output=True)
    assert result.returncode!=0


def test_restore_rejects_zip_path_traversal(tmp_path):
    root=tmp_path/'archive';root.mkdir();script=root/'restore_artifacts.py';script.write_bytes(SCRIPT.read_bytes())
    archive=root/'payload.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('../escape.txt','bad')
    data=archive.read_bytes();part=root/'part.bin';part.write_bytes(data);h=hashlib.sha256(data).hexdigest()
    record={'logical_path':'archives/example.zip','size':len(data),'sha256':h,'parts':[{'path':'part.bin','size':len(data),'sha256':h}]}
    (root/'artifact_manifest.json').write_text(json.dumps({'artifacts':[record]}),encoding='utf8')
    result=subprocess.run([sys.executable,str(script),'--destination',str(tmp_path/'dest'),'--extract'],capture_output=True)
    assert result.returncode!=0 and not (tmp_path/'escape.txt').exists()
