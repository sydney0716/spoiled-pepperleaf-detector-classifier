import redis
import paho.mqtt.client as mqtt
import time
import logging
import os
import json
import cv2
import numpy as np
from logging.handlers import RotatingFileHandler
import vision_inference_module as vim


# Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
IMAGE_ROOT = "./images"

# Redis for state management
r = redis.Redis(host="localhost", port=6379, db=0)

# MQTT client
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)


# Logging configuration
LOG_DIR = os.path.join(os.getcwd(), "logging")
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "image_dispatcher_worker.log")
logger = logging.getLogger("image_dispatcher_worker")
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,
    backupCount=100,
    encoding="utf-8"
)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def update_state(esp_id, state):
    """
    Update ESP state in Redis and publish via MQTT.
    """
    key = f"status:{esp_id}"
    current_state = r.get(key)
    current_state = current_state.decode() if current_state else "CLEAR"

    if state == "ALERT":
        r.set(key, "ALERT")
        topic = f"farm/alert/{esp_id}"
        client.publish(topic)
        logger.info(f"[ALERT] Published to {topic}")
        return

    elif state == "NEARBY":
        if current_state == "ALERT":
            return
        
        r.set(key, "NEARBY")
        topic = f"farm/nearby/{esp_id}"
        client.publish(topic)
        logger.info(f"[NEARBY] Published to {topic}")
        return
        
    elif state == "CLEAR":
        if current_state == "NEARBY":
            logger.info(f"[CLEAR] Ignored due to existing NEARBY state")
            return  
        r.set(key, "CLEAR")
        topic = f"farm/clear/{esp_id}"
        client.publish(topic)
        logger.info(f"[CLEAR] Published to {topic}")
        return


def worker_loop(model_data):
    """
    Main worker loop that processes images from Redis queue.
    Performs detection and classification.
    Updates ESP state based on result.
    """
    while True:
        item = r.lpop("image_queue")
        if not item:
            time.sleep(0.1)
            continue

        data = json.loads(item)
        esp_id = str(data["esp_id"])
        img_bytes = bytes.fromhex(data["image"])

        # Image decoding
        try:
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error(f"[Worker] ESP {esp_id} - Failed to decode image bytes")
                continue
        except Exception as e:
            logger.error(f"[Worker] ESP {esp_id} - Image decoding error: {e}")
            continue
            
        logger.info(f"[Worker] ESP {esp_id} - Processing image")

        # AI inference
        result, annotated_frame = vim.detect_and_classify_image(frame, model_data)
        
        if result is None:
            logger.error(f"[Worker] ESP {esp_id} - Inference failed")
            continue
            
        # Save processed image
        folder = os.path.join(IMAGE_ROOT, esp_id)
        os.makedirs(folder, exist_ok=True)
 
        result_str = "spoiled" if result == 1 else "normal"
        filename = os.path.join(
            folder, 
            f"{int(time.time()*1000)}_{esp_id}_{result_str}.jpg"
        )

        cv2.imwrite(filename, annotated_frame) 
        logger.info(
            f"[Worker] ESP {esp_id} - Processed. "
            f"Result: {result} ({result_str}). Log saved: {filename}"
        )

        # Priority-based state update
        if result == 1:
            update_state(esp_id, "ALERT")

            esp_num = int(esp_id)
            for tid in range(esp_num - 2, esp_num + 3):
                if tid <= 0 or tid == esp_num:
                    continue
                update_state(str(tid), "NEARBY")
        else:
            update_state(esp_id, "CLEAR")


if __name__ == "__main__":
    logger.info("Waiting for jobs...")
    model_data = vim.load_models()
    worker_loop(model_data)
