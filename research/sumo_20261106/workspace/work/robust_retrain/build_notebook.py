from pathlib import Path
import nbformat as nb,zipfile,base64,hashlib
root=Path("R:/robust_retrain")
project=root/"T_Junction_RL"
out=Path.cwd()/"outputs"
archive=root/"robust_source.zip"
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for f in project.rglob("*"):
  if f.is_file() and "__pycache__" not in f.parts and "outputs" not in f.parts and f.suffix not in (".pyc",):
   z.write(f,Path("T_Junction_RL")/f.relative_to(project))
blob=archive.read_bytes();data=base64.b64encode(blob).decode();sha=hashlib.sha256(blob).hexdigest()
md=lambda s:nb.v4.new_markdown_cell(s)
code=lambda s:nb.v4.new_code_cell(s)
cells=[
md("""# SUMO 강화학습 재학습: 악조건과 센서 오차

다중 T자 교차로에서 신경망을 **처음부터 재학습**합니다. 5개 학습 지도, 2개 검증 지도와 그 어느 단계에도 쓰지 않는 새로운 시험 지도 2개를 사용합니다. 학습과 검증 수요 시드도 기존 실험과 분리했습니다. 학습 시드 5개 × 600회입니다.

매 결정에 ±15% 범위의 독립 관측 오차를 주고 대기 보호를 적용합니다. 보상에 오랜 무녹색 대기에 대한 페널티를 추가합니다. 검증용 선택 점수는 지연의 평균 50%와 상위 20% 평균 50%를 합칩니다. 보호 계층은 SUMO의 실제 대기 정보를 보므로 현실 센서 유실에 대한 안전 보증은 아닙니다.

학습을 완료하면 새 시험 지도에서 기존 모델 1개, 재학습 모델 5개, 대기열 기준을 동일 수요·동일 보호 계층으로 비교합니다. 정상 관측과 ±15% 관측 오차를 각각 평가합니다. 전체 센서 단절, PiCar, 실제 도로는 시험 범위가 아닙니다. 동일한 개선율은 보장하지 않습니다."""),
code(f"""import base64, hashlib, io, zipfile, pathlib, os, sys
blob=base64.b64decode({data!r})
assert hashlib.sha256(blob).hexdigest()=={sha!r}
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    for x in z.infolist():
        assert (pathlib.Path('/content')/x.filename).resolve().is_relative_to(pathlib.Path('/content').resolve())
    z.extractall('/content')
os.chdir('/content/T_Junction_RL')
sys.path.insert(0,os.getcwd())
print('재학습 소스 SHA-256:',hashlib.sha256(blob).hexdigest())
"""),
code("""import subprocess,sys
subprocess.run([sys.executable,'-m','pip','install','-q',
 'eclipse-sumo==1.28.0','sumo-data==1.28.0','traci==1.28.0','sumolib==1.28.0','libsumo==1.28.0',
 'gymnasium>=1.0,<2','PyYAML>=6,<7','scipy>=1.11,<2','pandas>=2.1,<4'],check=True)
import libsumo, torch
print('SUMO:',libsumo.getVersion(),'PyTorch:',torch.__version__,'CUDA:',torch.cuda.is_available())
"""),
md("## Colab 예비 학습: 3회"),
code("""from pathlib import Path
import subprocess,sys,json
SMOKE=Path('/content/robust_study_smoke')
if not (SMOKE/'training_status.json').exists():
    subprocess.run([sys.executable,'-m','src.train','--mode','smoke','--output',str(SMOKE),
      '--training-seed','42','--algorithm','rainbow_lite','--control-mode','direct','--backend','libsumo','--device','cpu'],check=True)
status=json.loads((SMOKE/'training_status.json').read_text())
assert status['complete'] and status['updates']>0 and 'best_validation_robust_score' in status
print('예비 실행 결과:',status)
"""),
md("## Drive 연결 및 기존 결과와 분리된 새 폴더"),
code("""from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
ROBUST_DRIVE=Path('/content/drive/MyDrive/SUMO_Robust_Retrain_20261104_v1')
ROBUST_DRIVE.mkdir(parents=True,exist_ok=True)
os.environ['ROBUST_DRIVE']=str(ROBUST_DRIVE)
print('결과 저장:',ROBUST_DRIVE)
"""),
md("## 시드 5개 재학습. 5회마다 재개 체크포인트를 Drive에 저장합니다."),
code("""from src.train import train
import shutil,json
for seed in (42,43,44,45,46):
    local=Path('/content/robust_train_runs')/f'seed{seed}'
    mirror=ROBUST_DRIVE/f'seed{seed}'
    if not (local/'checkpoints/training_checkpoint.pt').exists() and (mirror/'checkpoints/training_checkpoint.pt').exists():
        shutil.copytree(mirror,local,dirs_exist_ok=True)
    cp=local/'checkpoints/training_checkpoint.pt'
    if cp.exists():
        train('full',local,resume=cp,mirror=mirror)
    else:
        train('full',local,training_seed=seed,algorithm='rainbow_lite',control_mode='direct',
              backend='libsumo',device='cpu',mirror=mirror)
    shutil.copy2(local/'training_status.json',mirror/'training_status.json')
    print('DONE',seed,(local/'training_status.json').read_text())
"""),
md("## 새로운 시험지도에서 896회 평가. 재실행 시 이미 저장된 회차는 건너뜁니다."),
code("""import os,subprocess,sys
os.environ.pop('ROBUST_EVAL_SMOKE',None)
subprocess.run([sys.executable,'-u','-m','scripts.robust_evaluate'],check=True)
""")]
n=nb.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
nb.validate(n);path=out/"T_Junction_Colab_Robust_Retrain.ipynb";nb.write(n,path)
print(path,"source zip bytes",len(blob),"SHA256",sha)
