import paho.mqtt.client as mqtt
import time
import logging
import os

# MQTT 브로커 설정
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# 로깅 설정
LOG_DIR = os.path.join(os.getcwd(), "logging")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "capture_scheduler.log")

logger = logging.getLogger("CaptureScheduler")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file, encoding="utf-8")
formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# MQTT 클라이언트 생성
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

def publish_capture(interval=5):
    """주기적으로 farm/capture 메시지를 발행"""
    while True:
        client.publish("farm/capture", "1")
        logger.info("[Capture] Published farm/capture")
        time.sleep(interval)

if __name__ == "__main__":
    logger.info("Capture Scheduler started...")
    publish_capture(interval=10)
