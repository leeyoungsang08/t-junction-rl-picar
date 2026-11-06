# SUMO T자 교차로 강화학습: 2026-11-03~05 연구 기록

이 디렉터리는 기존 프로젝트의 후속 연구 자료를 보존한다. 기존 PiCar·신호 코드와 모델은 교체하지 않았다. 실험의 주제는 **차량 조향이 아니라 교차로 신호 제어**다. SUMO 시뮬레이션 결과이며 실제 PiCar·실도로 성능이나 안전성을 입증하지 않는다.

## 먼저 볼 문서

- [전체 진행·결과 요약](RESEARCH_SUMMARY_KO.md)
- [원본 파일과 보존 자료 목록](ARTIFACT_INDEX_KO.md)
- [PiCar 연결 전 센서·재학습 시험](reports/SUMO_PiCar_연결전_실험보고서.md)
- [balanceA 교차로 안정화와 장시간 대기 시험](reports/SUMO_balanceA_안정화_재학습_보고서.md)
- [보상 변경 단일 실험](reports/SUMO_보상_단일변경_재학습_기록.md)
- [Colab 600회 학습과 1,040회 평가 해석](reports/T_Junction_Colab_Result_Interpretation_KR.md)
- [PiCar 실측·연결 준비](reports/SUMO_PiCar_실측_및_연결_준비.md)

## 현재까지 확인된 결론

1. SUMO에서 신호 제어 정책의 강화학습과 여러 재학습이 실제로 진행됐다. 학습 횟수나 신경망 업데이트만으로 성능 향상을 단정할 수 없다.
2. 새 지도·수요 시험에서 기존 모델과 후보의 우열은 교통량과 센서 조건에 따라 달랐다. 새 모델이 모든 상황에서 우수하다고 확인되지 않아 기존 프로젝트의 기본 모델을 교체하지 않았다.
3. balanceA 단일 지도에 대한 후속 재학습 6회는 각 180회 에피소드를 완료했으나, 사전 정의한 조건별 검증 기준을 통과한 학습 후 체크포인트가 없었다.
4. 방향별 장시간 무녹색 감시를 추가하면 관측된 최장 대기가 줄어드는 사례가 있었으나 일부 교통 상황의 지연이 악화됐다. 차량별 대기 상한이나 실제 안전을 보장하지 않는다.
5. 센서 누락·지연 및 통신 중단은 합성 가정이다. SUMO 설정에서 자동 텔레포트를 끈 시험의 텔레포트 0건은 교착이 없다는 뜻이 아니다.

## 파일과 재현

| 경로 | 내용 |
|---|---|
| `reports/` | 각 단계의 한국어 결과·변경 기록 |
| `figures/` | SUMO 시계열과 비교 그래프 |
| `notebooks/` | 로컬에 보존된 Colab 노트북 사본. 아래의 최신 Drive 노트북과 동일하다고 가정하지 말 것 |
| `workspace/work/` | 실험·검증 스크립트, 테스트, 분리된 프로젝트 소스 및 기존 모델 |
| `artifacts/` | 모델, 로그, 입력 지도·경로, 결과 CSV 및 원본 자료의 분할 보존본 |
| `artifact_manifest.json` | 원본별 크기·SHA-256과 각 분할 조각의 크기·SHA-256 |
| `restore_artifacts.py` | 검증·복원 도구. 기존에 다른 내용의 파일이 있으면 덮어쓰지 않음 |

GitHub 단일 파일 크기 제한을 피하기 위해 자료를 24 MiB 이하 조각으로 나눴다. 이 폴더에서 모든 조각과 원본 해시를 검사하려면:

```bash
python restore_artifacts.py --verify-only
```

복원과 결과 디렉터리 압축 해제를 함께 하려면:

```bash
python restore_artifacts.py --destination workspace --extract
```

그러면 `workspace/outputs/`에 원래 로컬 결과가 복원되고, `workspace/archives/`에는 결과 디렉터리 보존용 ZIP이 남는다. 원본 첨부 자료는 `workspace/inputs/`에 복원된다. 기존 파일을 덮어쓰지 않으므로 새 디렉터리에서 실행한다. Windows에서 긴 경로 지원이 필요할 수 있다. GitHub 화면에서 `.bin` 조각을 직접 실행하는 방식이 아니다.

로컬 결과의 체크포인트·개별 에피소드 로그를 포함한 모든 `outputs/` 디렉터리와 사용자에게 전달된 원본 결과 파일을 분할 보존한다. 다만 이 기록 시점에 **Google Drive에만 남은 원본 학습·평가 로그와 최신 Colab 노트북 전체는 이 저장소로 내려받아 검증하지 못했다.** Drive 링크가 있는 보고서는 해당 외부 자료의 위치를 기록한 것이며 GitHub 내 사본을 뜻하지 않는다.

Colab 원본: [다중 지도 학습 노트북](https://colab.research.google.com/drive/15Ul3O5c207CLPgTDBsnsWDgZq3E6tpEo) · [후속 재학습 노트북](https://colab.research.google.com/drive/1qOAS8Gi0-qMnEGi5IQGsYFHSdM2DwhmI). 접근은 해당 Google 계정의 권한에 따른다. Colab 화면에서 저장 충돌이 표시된 경우 로컬 화면과 Drive에 저장된 내용이 다를 수 있다.

로컬 최신 시험 환경은 Windows / Python 3.11.9 / SUMO 1.28.0 / CPU PyTorch 2.15.0이다. 각 실행의 입력과 소프트웨어 버전은 개별 manifest를 따른다. 출처·분할을 검증하는 것과 다른 환경에서 전 회차를 독립적으로 재실행하는 것은 다르다.
