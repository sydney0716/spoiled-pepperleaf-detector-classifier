import redis
import paho.mqtt.client as mqtt
import time
import logging
import os
import json
import cv2
import numpy as np
from logging.handlers import RotatingFileHandler
import raspberry_pi_improved as rp
# ------------------------
# Config
# ------------------------
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
IMAGE_ROOT = "./images"

# Redis 연결 (상태 관리)
r = redis.Redis(host="localhost", port=6379, db=0)

# MQTT 연결
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# ------------------------
# Logging 설정
# ------------------------
LOG_DIR = os.path.join(os.getcwd(), "logging")
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "worker.log")

logger = logging.getLogger("AIWorker")
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=100, encoding="utf-8")
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# ------------------------
# Publish Helper
# ------------------------
def update_state(esp_id, state):
    """
    ESP 상태를 Redis에 저장하고 MQTT 발행. (우선순위: ALERT > NEARBY > CLEAR)
    """
    key = f"status:{esp_id}"
    current_state = r.get(key)
    current_state = current_state.decode() if current_state else "CLEAR" # 기본 상태 CLEAR

    # 1. 상태 전이 규칙 적용 (핵심)
    if state == "ALERT":
        # ALERT는 항상 최우선.
        r.set(key, "ALERT")
        topic = f"farm/alert/{esp_id}"
        client.publish(topic)
        logger.info(f"[ALERT] Published to {topic}")
        return

    elif state == "NEARBY":
        # NEARBY는 ALERT 상태일 때만 무시됩니다.
        if current_state == "ALERT":
            return
        
        r.set(key, "NEARBY")
        topic = f"farm/nearby/{esp_id}"
        client.publish(topic)
        logger.info(f"[NEARBY] Published to {topic}")
        return
        
    elif state == "CLEAR":
        # CLEAR는 NEARBY 상태일 때 무시됩니다.
        if current_state == "NEARBY":
            logger.info(f"[CLEAR] Ignored")
            return

        # CLEAR로 업데이트하고 발행
        else:
            r.set(key, "CLEAR")
            topic = f"farm/clear/{esp_id}"
            client.publish(topic)
            logger.info(f"[CLEAR] Published to {topic}")
        return

# ------------------------
# Worker Loop
# ------------------------
def worker_loop(model_data):
    while True:
        item = r.lpop("image_queue")
        if not item:
            time.sleep(0.1)
            continue

        data = json.loads(item)
        esp_id = str(data["esp_id"])
        img_bytes = bytes.fromhex(data["image"])

        try:
            # 바이트를 NumPy 배열로 변환
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            # NumPy 배열을 OpenCV 프레임으로 디코딩 (메모리 I/O)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error(f"[Worker] ESP {esp_id} - Failed to decode image bytes, skipping...")
                continue
        except Exception as e:
            logger.error(f"[Worker] ESP {esp_id} - Image decoding error: {e}, skipping...")
            continue
            
        logger.info(f"[Worker] ESP {esp_id} - Processing image (Memory-based)")

        # ----------------------------------------------------
        # 2. AI 처리 (수정된 메모리 기반 함수 호출)        
        # ----------------------------------------------------
        result, annotated_frame = rp.detect_and_classify(frame, model_data)
        
        if result is None:
            logger.error(f"[Worker] ESP {esp_id} - Inference failed, skipping...")
            continue
            
        # ----------------------------------------------------
        # 3. 결과 프레임 디스크 저장 (로깅 목적으로 처리 후 실행)
        # ----------------------------------------------------
        folder = os.path.join(IMAGE_ROOT, esp_id)
        os.makedirs(folder, exist_ok=True)
        # 파일 이름에 결과 분류(정상/병충해)를 포함하여 로깅 편의성 확보
        result_str = "spoiled" if result == 1 else "normal"
        filename = os.path.join(folder, f"{int(time.time()*1000)}_{esp_id}_{result_str}.jpg")

        # 시각화된 프레임을 저장
        cv2.imwrite(filename, annotated_frame) 
        
        logger.info(f"[Worker] ESP {esp_id} - Processed. Result: {result} ({result_str}). Log saved: {filename}")

        # ------------------------
        # 3. 우선순위 기반 상태 발행
        # ------------------------
        if result == 1:
            # 1순위: Alert → ESP 자신
            update_state(esp_id, "ALERT")
            # 주변 ±2 ESP: Nearby
            esp_num = int(esp_id)
            for tid in range(esp_num - 2, esp_num + 3):
                if tid <= 0 or tid == esp_num:
                    continue
                update_state(str(tid), "NEARBY")
        else:
            # 3순위: Clear → Alert/Nearby가 아닌 경우만 적용
            update_state(esp_id, "CLEAR")


if __name__ == "__main__":
    logger.info("AI Worker started. Waiting for jobs...")
    model_data = rp.load_models()
    worker_loop(model_data)
