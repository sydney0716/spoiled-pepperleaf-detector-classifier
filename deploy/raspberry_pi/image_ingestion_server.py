from flask import Flask, request
import redis
import json

# Redis configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# Image upload endpoint
app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload_image():
    """
    HTTP endpoint for receiving JPEG images from ESP devices.
    The image is stored in Redis queue as a hex-encoded string.

    Headers:
        ESP-ID: Unique Identifier for each ESP32 device.

    Returns:
        200 OK on success, 400 if ESP-ID is missing.
    """
    esp_id = request.headers.get("ESP-ID")
    if not esp_id:
        return "Missing ESP-ID", 400

    img_bytes = request.data

    # Push encoded image and metadata to Redis queue
    r.rpush("image_queue", json.dumps({
        "esp_id": esp_id,
        "image": img_bytes.hex()
    }))

    return "200 OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)