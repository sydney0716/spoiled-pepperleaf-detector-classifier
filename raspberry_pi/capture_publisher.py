import paho.mqtt.client as mqtt
import time
import logging
import os

# MQTT configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# Logging configuration
LOG_DIR = os.path.join(os.getcwd(), "logging")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "capture_publisher.log")

logger = logging.getLogger("capture_publisher")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file, encoding="utf-8")
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# MQTT client
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

def publish_capture(interval=10):
    """Periodically publishes a capture trigger message to the MQTT broker."""
    while True:
        client.publish("farm/capture", "1")
        logger.info("[Capture] Published capture trigger to farm/capture")
        time.sleep(interval)


if __name__ == "__main__":
    publish_capture(interval=10)
