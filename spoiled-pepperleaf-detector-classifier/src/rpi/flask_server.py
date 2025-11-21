from flask import Flask, request
import redis
import json

REDIS_HOST = "localhost"
REDIS_PORT = 6379

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload_image():
    esp_id = request.headers.get("ESP-ID")
    if not esp_id:
        return "Missing ESP-ID", 400

    img_bytes = request.data
    # Redis 큐에 ESP-ID + 이미지 바이트(hex)만 push
    r.rpush("image_queue", json.dumps({
        "esp_id": esp_id,
        "image": img_bytes.hex()
    }))

    return "200 OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
