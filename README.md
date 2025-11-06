# spoiled-pepperleaf-detector-classifier
YOLO-based Korean pepper leaf detection and ResNet18 disease classification for edge deployment
---

## Code Description
1. capture_scheduler.py
   - MQTT를 이용해 `farm/capture` 토픽에 주기적으로 메시지를 발행.
   - ESP32 장치들은 이 메시지를 수신받으면 촬영하게 됨

2. esp32_final.ino
   - ESP32 쪽 아두이노 코드로, 카메라로 사진을 캡쳐하고 Flask 서버로 JPEG 이미지를 HTTP POST로 업로드
   - MQTT를 통해 중앙 서버로부터 LED 색상 토픽을 수신 & LED 점등
  
3. flask_server.py
   - ESP32로부터 업로드된 JPEG 이미지를 수신받고, 각 이미지에 포함된 ESP-ID와 이미지 바이트를 Redis에 JSON으로 저장
   - 실시간 AI 처리를 위해 디스크 대신 Redis에 저장
  
4. ai_worker.py
   - Redis 큐에서 이미지를 꺼내서 raspberry_pi_improved.py로 분석.
   - 분석 결과에 따라 LED 색상을 결정하는 MQTT 토픽 발행
   - 결과 이미지를 로컬 폴더에 저장하고 로그 기록
  
5. raspberry_pi_improved.py
   - AI 추론 모듈로, 정상/병충해 여부 판정하여 결과 반환
   - 혹시 모듈 수정했으면 여기 구조 바꿀 것. 처음 보내준 결과 매트리스 크기 안 맞아서 수정함.
