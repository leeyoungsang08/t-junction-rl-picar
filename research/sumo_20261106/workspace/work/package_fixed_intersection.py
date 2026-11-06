from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/SUMO_balanceA_안정화_재학습_결과.zip'
SOURCE=ROOT/'work/robust_retrain/T_Junction_RL_reward_ablation'
WORK=('anchored_dqn.py','fixed_intersection_train.py','fixed_intersection_gate.py','fixed_intersection_evaluate.py',
      'fixed_intersection_watchdog.py','audit_fixed_intersection.py','analyze_watchdog.py','plot_fixed_intersection.py','plot_watchdog.py',
      'replay_verify_fixed_intersection.py','replay_verify_watchdog.py',
      'test_anchored_dqn.py','test_fixed_intersection.py','pre_picar_study.py',
      'pre_picar_training.py','pre_picar_balanced_evaluate.py')
TRAIN=ROOT/'outputs/SUMO_balanceA_안정화_20261105'
DATA=('SUMO_balanceA_기준정책_안전판_검증','SUMO_balanceA_독립시험_20261105','SUMO_balanceA_장시간대기_추가시험')
README='''# 특정 교차로 강화학습 실험

이 패키지는 SUMO의 balanceA 단일 교차로 연구 기록이다. 실제 PiCar, 실제 센서 또는
실제 도로 성능과 안전성을 입증하지 않는다. 모든 교통 상황의 개선 보장은 없다.

원본 모델, 현재 코드, 고정 교차로, 학습 체크포인트·검증 CSV, 검증 및 시험의 입력과
원본 압축 로그를 포함한다. 학습 에피소드별 원본과 36개 검증 시뮬레이션별 원본은
용량 때문에 묶음에서 제외했으며 이 PC의 학습 결과 폴더에 보존돼 있다.

실험 환경: Windows, Python 3.11.9, SUMO 1.28.0, CPU PyTorch 2.15.0.
각 manifest에는 입력 SHA-256, 버전, 조건 및 모델 선택 기준이 기록돼 있다.
같은 환경에서 예: python work/audit_fixed_intersection.py
단위 시험 예: python -m pytest -q work/test_anchored_dqn.py work/test_fixed_intersection.py

실행 재개 도구는 기존 파일의 해시를 확인하고 유효한 기존 결과를 재사용한다.
이는 독립 재실행이 아니다. 재현 실험은 별도 결과 폴더와 동일한 입력·환경을 써야 한다.
'''
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    z.writestr('README_재현.md',README)
    for n in WORK:z.write(ROOT/'work'/n,'work/'+n)
    z.write(ROOT/'work/robust_retrain/old_seed42.pt','work/robust_retrain/old_seed42.pt')
    for part in ('src','configs','maps'):
        for p in (SOURCE/part).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT))
    z.write(SOURCE/'requirements.txt',(SOURCE/'requirements.txt').relative_to(ROOT))
    for p in (TRAIN/'inputs').rglob('*'):
        if p.is_file():z.write(p,p.relative_to(ROOT))
    for n in ('manifest.json','audit_summary.json'):
        p=TRAIN/n
        if p.exists():z.write(p,p.relative_to(ROOT))
    for run in TRAIN.glob('*seed*'):
        if run.is_dir():
            for p in run.iterdir():
                if p.is_file():z.write(p,p.relative_to(ROOT))
    for dirname in DATA:
        for p in (ROOT/'outputs'/dirname).rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT))
    for p in (ROOT/'outputs/SUMO_PiCar_전_균형선택_새시험/inputs/maps/balanceA').glob('*'):
        if p.is_file():z.write(p,p.relative_to(ROOT))
    prior=ROOT/'outputs/SUMO_PiCar_전_균형선택_재학습/seed43/best.pt'
    z.write(prior,prior.relative_to(ROOT))
    for n in ('SUMO_balanceA_안정화_재학습_보고서.md','SUMO_balanceA_조건별_비교.png','SUMO_balanceA_대기와지연_비교.png'):
        p=ROOT/'outputs'/n
        if p.exists():z.write(p,p.relative_to(ROOT))
print(OUT,OUT.stat().st_size)
