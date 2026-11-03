# 중앙 UDP 통신

```text
상부 관측 + 출발 요청 → 상태 집계 → DQN KEEP/SWITCH → 신호 상태 머신
PiCar 센서 → UDP 상태·이벤트 → 중앙 원장 → 점유 확인 → UDP 진입 허가
라인 추종 → cmd_vel_raw → Pi 로컬 게이트 → cmd_vel → 모터 watchdog
```

현시 전환과 차량 한 대에 대한 진입 허가는 별도 결정이다. 처음에는 녹색이고 교차로 점유가 비어 있을 때 해당 방향 맨 앞 차량 한 대를 예약한다.

## PC↔Pi 모터 없는 시험

두 장치를 실험 LAN에 연결한다. Windows의 `ipconfig`, Pi의 `hostname -I`로 주소를 확인한다. 아래 192.168.50.10은 중앙 PC의 실제 LAN IPv4로 바꾼다. 공유기의 장치 격리와 PC의 실험 LAN UDP 수신을 점검한다. 포트 5005는 순수 왕복 확인, 5006은 상태 실습용이다. [Raspberry Pi 원격 접속 안내](https://www.raspberrypi.com/documentation/computers/remote-access.html)

```powershell
# PC에서 첫 왕복 시험
python outputs/budget_check_20261103/picar_lan_probe.py listen --host 192.168.50.10 --port 5005

# 위 시험을 끝낸 뒤 중앙 상태 수신기
python picar_lab/status_link.py server --host 192.168.50.10 --port 5006 --config picar_lab/config/lab.example.json --log lab_run/central_status.jsonl
```

Pi로 `picar_lan_probe.py`와 `status_link.py`를 복사한다. 두 파일은 Python 3 표준 라이브러리만 사용한다. 모터 전원을 분리한 상태에서 실행한다.

```bash
python3 picar_lan_probe.py probe --host 192.168.50.10 --port 5005 --count 100
python3 status_link.py client --host 192.168.50.10 --port 5006 --car S1 --lane S
```

다른 Pi는 등록표에서 별도 ID와 방향을 선택한다. 성공하면 PC와 Pi에 APPROACH → ARRIVE → IN → EXIT 및 ACK가 표시된다. 이벤트는 실습 코드가 만든 가상 이벤트다. 같은 ID로 새 시험을 하려면 중앙 실습 서버를 재시작한다. 이 서버의 출력은 상태 접수 결과다.

## 실습 패킷과 현장 계약

```json
{"v":1,"type":"lab_status","car":"S1","boot":"1234567890abcdef1234567890abcdef","seq":2,"lane":"S","state":"ARRIVE","line_ok":true,"obstacle":false}
```

```json
{"v":1,"type":"lab_ack","session":"abcdef1234567890abcdef1234567890","car":"S1","boot":"1234567890abcdef1234567890abcdef","ack":2,"result":"accepted"}
```

- 실습에서는 boot가 차량 실행 세션, session이 중앙 실행 세션이다. seq가 증가한다. 같은 패킷을 재전송하면 상태를 두 번 반영하지 않는다. 건너뛴 순서, 이전 순서, 세션 충돌을 분류한다.
- 현장 STATUS에는 센서 신선도, 위치 구역, 정지 여부, 요청 ID, 출발 방향과 목적지를 넣는다. PERMIT에는 서버 세션, 차량 세션, 요청 seq, grant ID, TTL을 넣는다. PERMIT_ACK는 Pi의 적용·거절 결과다. 차량별 인증과 무결성 검증을 구현한 뒤 모터 게이트에 연결한다.
- TTL은 Pi의 **요청 발송 시각부터** 로컬 단조시계로 판단한다. 재전송과 중복 ACK는 허가를 연장하지 않는다. 기준 TTL·주기는 실제 지연과 제동 거리로 정한다. 모의 코드의 250ms는 시연값이다. [IETF UDP 지침](https://www.rfc-editor.org/rfc/rfc8085.html)
- 차량 원장은 APPROACH → ARRIVE → RESERVED → IN → EXIT를 기록한다. 서버가 허가를 보내면 RESERVED로 잠근다. ACK 누락이나 TTL 경과만으로 슬롯을 해제하지 않는다. 차량 전체가 출구를 벗어났다는 독립 관측을 기다린다. 재시작이나 관측 불명확 시 신규 진입을 막고 상태를 재확인한다.
- 신호 제어기가 최소·최대 녹색, 황색·전적색, 교차로 실제 점유를 확인한다. Pi는 장애물, 라인 유실, 만료 시 자체 정지를 우선한다.

로그에는 서버 세션, 차량 ID, 시퀀스, 상태, 수신 단조시각, 현시, 점유, 허가 ID, 발급·ACK·진입·출구 및 센서 유실을 남긴다. RTT와 만료는 같은 기기의 단조시계로 계산한다.

[SUMO 정책→UDP 재생 결과](../../outputs/budget_check_20261103/picar_dry_run/policy_to_udp.json) · [UDP 장애 시험 결과](../../outputs/budget_check_20261103/picar_dry_run/picar_loopback_result.json) · [기존 Notion 설계](https://app.notion.com/p/6149c197abc783a2868f81c943696784)

