from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]
out=root/'outputs'/'SUMO_PiCar_연결전_코드_결과.zip'
source=root/'work/robust_retrain/T_Junction_RL_reward_ablation'
work_names=('pre_picar_study.py','pre_picar_training.py','pre_picar_evaluate.py','pre_picar_balanced_training.py',
            'pre_picar_balanced_evaluate.py','test_outage_fallback.py','watchdog_study.py','tune_fixed_baseline.py',
            'analyze_pre_picar.py','analyze_fixed.py','plot_pre_picar.py','plot_signal_trace.py','picar_shadow_adapter.py',
            'replay_verify_pre_picar.py','test_metric_determinism.py',
            'test_pre_picar_study.py','test_pre_picar_training.py','test_blind_watchdog.py','test_picar_shadow_adapter.py')
study_dirs=('SUMO_PiCar_전_센서시험','SUMO_PiCar_전_재학습_비교','SUMO_PiCar_전_고정신호_조정',
            'SUMO_PiCar_전_통신중단_대응','SUMO_PiCar_전_신호감시_시험','SUMO_PiCar_전_균형선택_새시험',
            'SUMO_결과계산_결정성_수정')
train_dirs=('SUMO_PiCar_전_센서재학습','SUMO_PiCar_전_균형선택_재학습')
text='''# SUMO PiCar 연결 전 실험 패키지

이 파일은 실제 PiCar가 아닌 SUMO 실험과 신호 추천 시제품이다.
재학습은 기존 가중치에서 시작했고 모델 교체는 별도 결과 해석이 필요하다.
관측 방식 및 모든 물리 파라미터는 미측정 가정이다. 실장치에 연결하지 말 것.

기록된 환경: Windows, Python 3.11.9, SUMO/libsumo 1.28.0, CPU PyTorch 2.15.0,
NumPy 2.4.6, Pandas 3.1.0, Gymnasium 1.4.0. 정확한 패키지 사용 여부는
각 실험 manifest의 버전·입력 해시 검사를 확인한다. requirements.txt에는 버전 범위가 있다.

압축을 푼 디렉터리를 작업 폴더로 사용한다. work 아래 스크립트와 project src/maps/configs,
원본 모델, outputs 아래 원본 실행 로그·입력·선택 체크포인트를 포함한다.
학습의 모든 개별 에피소드 원본 로그는 용량을 줄이기 위해 이 ZIP에 포함하지 않았고,
이 컴퓨터의 outputs/SUMO_PiCar_전_센서재학습 및 균형선택_재학습에 남아 있다.

예: python -m pytest -q work/test_pre_picar_study.py work/test_pre_picar_training.py
예: python -m pytest -q work/test_blind_watchdog.py work/test_picar_shadow_adapter.py
예: python work/analyze_pre_picar.py
예: python work/analyze_fixed.py

기존 실행의 재개 코드는 이미 기록한 로그를 해시 검사 후 재사용한다. 결과 파일 재사용을
독립 재실행으로 주장하면 안 된다. 별도 전체 재실행을 하려면 새로운 보존 경로 및 환경에서
코드를 조정하고, 생성 입력과 실행 환경이 동일한지 명시적으로 검증해야 한다.
기기 연결 시 측정해야 할 사항은 outputs/SUMO_PiCar_실측_및_연결_준비.md를 참조한다.
'''
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    z.writestr('README_재현.md',text)
    for n in work_names:z.write(root/'work'/n,'work/'+n)
    z.write(root/'work/robust_retrain/old_seed42.pt','work/robust_retrain/old_seed42.pt')
    for part in ('src','configs','maps'):
        for p in (source/part).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                z.write(p,p.relative_to(root))
    z.write(source/'requirements.txt',(source/'requirements.txt').relative_to(root))
    for name in study_dirs:
        for p in (root/'outputs'/name).rglob('*'):
            if p.is_file():z.write(p,p.relative_to(root))
    for name in train_dirs:
        for seed in ('seed42','seed43','seed44'):
            for n in ('best.pt','manifest.json','status.json','training.csv','validation.csv','training_checkpoint.pt'):
                p=root/'outputs'/name/seed/n
                if p.is_file():z.write(p,p.relative_to(root))
    for n in ('SUMO_PiCar_연결전_실험보고서.md','SUMO_PiCar_실측_및_연결_준비.md','SUMO_PiCar_신호추천_시제품.py','SUMO_PiCar_전_센서별_비교.png','SUMO_PiCar_새교차로_신호변화.png'):
        p=root/'outputs'/n
        if p.exists():z.write(p,p.relative_to(root))
print(out, out.stat().st_size)
