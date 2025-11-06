// #include <Arduino.h>
// #include "esp_camera.h"
// #include <WiFi.h>
// #include <PubSubClient.h>
// #include "FS.h"
// #include "SD_MMC.h"
// #include "esp_heap_caps.h"

// // ===== WiFi =====
// const char* WIFI_SSID = "AndroidHotspotff_45_8f";
// const char* WIFI_PASS = "justin526";
// const char* SERVER_HOST = "192.168.195.209";
// const int SERVER_PORT = 5000;

// // ===== MQTT =====
// const char* MQTT_BROKER = "192.168.195.196";  // mqtt 서버
// const int MQTT_PORT = 1883;

// const char* ESP32_ID = "10";  // 고유 ESP32 ID
// const char* CAPTURE_TOPIC = "farm/capture";
// char ALERT_TOPIC[32];
// char NEARBY_TOPIC[32];
// char CLEAR_TOPIC[32];

// WiFiClient espClient;
// PubSubClient mqttClient(espClient);

// #define CAMERA_MODEL_AI_THINKER
// #define PWDN_GPIO_NUM     32
// #define RESET_GPIO_NUM    -1
// #define XCLK_GPIO_NUM      0
// #define SIOD_GPIO_NUM     26
// #define SIOC_GPIO_NUM     27
// #define Y9_GPIO_NUM       35
// #define Y8_GPIO_NUM       34
// #define Y7_GPIO_NUM       39
// #define Y6_GPIO_NUM       36
// #define Y5_GPIO_NUM       21
// #define Y4_GPIO_NUM       19
// #define Y3_GPIO_NUM       18
// #define Y2_GPIO_NUM        5
// #define VSYNC_GPIO_NUM    25
// #define HREF_GPIO_NUM     23
// #define PCLK_GPIO_NUM     22
// #define FLASH_LED_GPIO     4

// // LED PIN
// #define RED_LED_GPIO      12
// #define GREEN_LED_GPIO    13

// #define CAPTURE_INTERVAL_MS 5000

// // LED Config
// #define LED_CHANNEL_R 1
// #define LED_CHANNEL_G 2
// const int PWM_FREQ = 5000;
// const int PWM_RESOLUTION = 8; // 최소 8비트 이상

// // 주황색 구현을 위한 값 (CA 타입)
// const int RED_BRIGHTNESS = 0;
// const int GREEN_BRIGHTNESS = 180;

// void setLED_PWM(int redValue, int greenValue) {
//     ledcWrite(LED_CHANNEL_R, redValue);
//     ledcWrite(LED_CHANNEL_G, greenValue);
// }

// void setRed() { setLED_PWM(0, 255); }         
// void setGreen() { setLED_PWM(255, 0); }      
// void setOrange() { setLED_PWM(RED_BRIGHTNESS, GREEN_BRIGHTNESS); }
// void setYellow() { setLED_PWM(0, 0); }
// void setOff() { setLED_PWM(255, 255); }         

// // ===== Camera init =====
// bool initCamera() {
//     camera_config_t config;
//     config.ledc_channel = LEDC_CHANNEL_0;
//     config.ledc_timer   = LEDC_TIMER_0;
//     config.pin_d0       = Y2_GPIO_NUM;
//     config.pin_d1       = Y3_GPIO_NUM;
//     config.pin_d2       = Y4_GPIO_NUM;
//     config.pin_d3       = Y5_GPIO_NUM;
//     config.pin_d4       = Y6_GPIO_NUM;
//     config.pin_d5       = Y7_GPIO_NUM;
//     config.pin_d6       = Y8_GPIO_NUM;
//     config.pin_d7       = Y9_GPIO_NUM;
//     config.pin_xclk     = XCLK_GPIO_NUM;
//     config.pin_pclk     = PCLK_GPIO_NUM;
//     config.pin_vsync    = VSYNC_GPIO_NUM;
//     config.pin_href     = HREF_GPIO_NUM;
//     config.pin_sccb_sda = SIOD_GPIO_NUM;
//     config.pin_sccb_scl = SIOC_GPIO_NUM;
//     config.pin_pwdn     = PWDN_GPIO_NUM;
//     config.pin_reset    = RESET_GPIO_NUM;
//     config.xclk_freq_hz = 20000000;
//     config.pixel_format = PIXFORMAT_JPEG;

//     if(psramFound()){
//         config.frame_size = FRAMESIZE_VGA;
//         config.jpeg_quality = 12;
//         config.fb_count = 2;
//     } else {
//         config.frame_size = FRAMESIZE_QVGA;
//         config.jpeg_quality = 15;
//         config.fb_count = 1;
//     }

//     return esp_camera_init(&config) == ESP_OK;
// }

// // ===== WiFi =====
// void connectWiFi() {
//     WiFi.mode(WIFI_STA);
//     WiFi.begin(WIFI_SSID, WIFI_PASS);
//     while(WiFi.status() != WL_CONNECTED) {
//         delay(500);
//         Serial.print(".");
//     }
//     Serial.printf("\nWiFi connected, IP: %s\n", WiFi.localIP().toString().c_str());
// }

// // ===== SD 초기화 =====
// bool initSD() {
//     if(!SD_MMC.begin()){
//         Serial.println("SD_MMC Mount Failed");
//         return false;
//     }
//     return true;
// }

// // ===== JPEG HTTP 전송 =====
// bool postJpeg(const uint8_t* buf, size_t len) {
//     WiFiClient client;
//     if(!client.connect(SERVER_HOST, SERVER_PORT)) {
//       setYellow();
//       return false;
//     }
//     client.printf("POST /upload HTTP/1.1\r\n");
//     client.printf("Host: %s\r\n", SERVER_HOST);
//     client.printf("Content-Type: image/jpeg\r\n");
//     client.printf("Content-Length: %d\r\n", len);
//     client.printf("ESP-ID: %s\r\n", ESP32_ID);
//     client.print("Connection: close\r\n\r\n");
//     client.write(buf, len);
//     client.flush();

//     String resp;
//     uint32_t start = millis();
//     while(millis()-start < 8000){
//         while(client.available()) resp += char(client.read());
//         if(!client.connected()) break;
//         delay(10);
//     }
//     client.stop();

//     if (resp.indexOf("200 OK") > 0) {
//         setLEDByState(); // 성공: LED 상태를 현재 currentState(ALERT/NEARBY/CLEAR)에 맞게 복구
//         return true;
//     }

//     // 실패: 실패 상태(Yellow)를 표시하고, 이전 경고 상태를 복원하지 않습니다.
//     // 실패가 발생했음을 사용자에게 명확히 알려야 합니다.
//     setYellow(); 
//     return false;
// }

// // ===== SD 저장 =====
// void saveToSD(const uint8_t* buf, size_t len){
//     char filename[64];
//     snprintf(filename, sizeof(filename), "/failed_%u_%06X.jpg", millis(), (uint32_t)(ESP.getEfuseMac() & 0xFFFFFF));
//     File file = SD_MMC.open(filename, FILE_WRITE);
//     if(file){
//         file.write(buf, len);
//         file.close();
//         Serial.printf("[SD] Saved %s\n", filename);
//     } else {
//         Serial.println("[SD] Save failed");
//     }
// }

// // ===== SD 재전송 =====
// void retrySD() {
//     File root = SD_MMC.open("/");
//     File file = root.openNextFile();
//     while(file){
//         if(!file.isDirectory()){
//             File f = SD_MMC.open(file.name());
//             if(!f){ 
//               file = root.openNextFile();
//               continue;
//             }

//             WiFiClient client;
//             if(client.connect(SERVER_HOST, SERVER_PORT)){
//                 client.printf("POST /upload HTTP/1.1\r\n");
//                 client.printf("Host: %s\r\n", SERVER_HOST);
//                 client.printf("Content-Type: image/jpeg\r\n");
//                 client.printf("Content-Length: %d\r\n", f.size());
//                 client.printf("ESP-ID: %s\r\n", ESP32_ID);
//                 client.print("Connection: close\r\n\r\n");

//                 uint8_t buf[4096];
//                 while(f.available()){
//                     size_t n = f.read(buf, sizeof(buf));
//                     client.write(buf, n);
//                 }
//                 client.flush();

//                 String resp;
//                 uint32_t start = millis();
//                 while(millis()-start < 8000){
//                     while(client.available()) resp += char(client.read());
//                     if(!client.connected()) break;
//                     delay(10);
//                 }
//                 client.stop();

//                 if(resp.indexOf("200 OK")>0){
//                     SD_MMC.remove(file.name());
//                     Serial.printf("[SD] Successfully re-uploaded %s\n", file.name());
//                 }
//             }
//             f.close();
//         }
//         file = root.openNextFile();
//     }
// }

// // ===== LED 상태 관리 =====
// String currentState = "CLEAR"; // 기본 상태

// void setLEDByState() {
//     if(currentState == "ALERT") setRed();
//     else if(currentState == "NEARBY") setOrange();
//     else setGreen();
// }

// // ===== MQTT Callback =====
// void callback(char* topic, byte* payload, unsigned int length){
    
//     // String 비교는 느리므로, 토픽 주소의 시작 포인터만 비교해도 됨 (startsWith)
//     String received_topic = String(topic);
    
//     // 1. ALERT 처리 (최우선)
//     if(received_topic.startsWith("farm/alert/")){
//         currentState = "ALERT";
//         setLEDByState();
//     }
//     // 2. NEARBY 처리 (ALERT가 아닐 때만 적용)
//     else if(received_topic.startsWith("farm/nearby/")){
        
//         if(currentState != "ALERT"){
//             currentState = "NEARBY";
//             setLEDByState();
//         }
//     }
//     // 3. CLEAR 처리 (ALERT, NEARBY가 아닐 때만 적용)
//     else if(received_topic.startsWith("farm/clear/")){
//         if(currentState != "ALERT" && currentState != "NEARBY"){
//             currentState = "CLEAR";
//             setLEDByState();
//         }
//     }

//     // 4. Capture 요청 처리
//     if(received_topic == CAPTURE_TOPIC){
//         camera_fb_t* fb = esp_camera_fb_get();
//         if(fb){
//             if(!postJpeg(fb->buf, fb->len)){
//                 saveToSD(fb->buf, fb->len);
//                 // **postJpeg()가 실패하면 Yellow 상태가 됩니다.**
//                 // **여기서는 추가적인 LED 상태 복원 코드를 넣지 않습니다.** // 실패가 발생했음을 Yellow로 보여주고, 다음번 MQTT 루프에서 
//                 // 상태(ALERT/NEARBY/CLEAR) 메시지가 오기를 기다려야 합니다.
//             }
//             esp_camera_fb_return(fb);
//         }
//     }
// }

// // ===== MQTT 연결 =====
// void connectMQTT(){
//     while(!mqttClient.connected()){
//         if(mqttClient.connect(ESP32_ID)){
//             snprintf(ALERT_TOPIC,sizeof(ALERT_TOPIC),"farm/alert/%s",ESP32_ID);
//             snprintf(NEARBY_TOPIC,sizeof(NEARBY_TOPIC),"farm/nearby/%s",ESP32_ID);
//             snprintf(CLEAR_TOPIC,sizeof(CLEAR_TOPIC),"farm/clear/%s",ESP32_ID);

//             mqttClient.subscribe(ALERT_TOPIC);
//             mqttClient.subscribe(NEARBY_TOPIC);
//             mqttClient.subscribe(CLEAR_TOPIC);
//             mqttClient.subscribe(CAPTURE_TOPIC);
//         } else delay(1000);
//     }
// }

// // ===== Setup =====
// void setup(){
//     Serial.begin(115200);

//     pinMode(FLASH_LED_GPIO,OUTPUT);
//     digitalWrite(FLASH_LED_GPIO,LOW);

//     // Legacy Code
//     // ledcSetup(LED_CHANNEL_R, PWM_FREQ, PWM_RESOLUTION);
//     // ledcAttachPin(RED_LED_GPIO, LED_CHANNEL_R);
//     //ledcSetup(LED_CHANNEL_G, PWM_FREQ, PWM_RESOLUTION);
//     //ledcAttachPin(GREEN_LED_GPIO, LED_CHANNEL_G);

    

//     setOff();

//     connectWiFi();
//     setGreen();

//     mqttClient.setServer(MQTT_BROKER,MQTT_PORT);
//     mqttClient.setCallback(callback);

//     if(!initCamera()){ Serial.println("Camera init failed"); ESP.restart(); }
//     if(!initSD()){ Serial.println("SD init failed"); }

//     ledcAttachChannel(RED_LED_GPIO, PWM_FREQ, PWM_RESOLUTION, LED_CHANNEL_R);
//     ledcAttachChannel(GREEN_LED_GPIO, PWM_FREQ, PWM_RESOLUTION, LED_CHANNEL_G);
// }

// // ===== Loop =====
// void loop(){
//     if(!mqttClient.connected()) connectMQTT();
//     mqttClient.loop();
//     retrySD();  // SD에 남은 사진 재전송
//     delay(10);
// }




#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>
#include <PubSubClient.h>
// #include "FS.h"      // 제거
// #include "SD_MMC.h"  // 제거
#include "esp_heap_caps.h"

// ===== WiFi =====
const char* WIFI_SSID = "AndroidHotspotff_45_8f";
const char* WIFI_PASS = "justin526";
const char* SERVER_HOST = "192.168.195.196";
const int SERVER_PORT = 5000;

// ===== MQTT =====
const char* MQTT_BROKER = "192.168.195.196";  // mqtt 서버
const int MQTT_PORT = 1883;

const char* ESP32_ID = "10";  // 고유 ESP32 ID
const char* CAPTURE_TOPIC = "farm/capture";
char ALERT_TOPIC[32];
char NEARBY_TOPIC[32];
char CLEAR_TOPIC[32];

WiFiClient espClient;
PubSubClient mqttClient(espClient);

#define CAMERA_MODEL_AI_THINKER
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define FLASH_LED_GPIO     4

// LED PIN
#define RED_LED_GPIO      12
#define GREEN_LED_GPIO    13

#define CAPTURE_INTERVAL_MS 5000

// LED Config
#define LED_CHANNEL_R 1
#define LED_CHANNEL_G 2
const int PWM_FREQ = 5000;
const int PWM_RESOLUTION = 8; // 최소 8비트 이상

ledcAttach(RED_LED_GPIO, PWM_FREQ, PWM_RESOLUTION);
ledcAttach(GREEN_LED_GPIO, PWM_FREQ, PWM_RESOLUTION);

// (CA 타입 -> CA라 255 or 256가 OFF / 0이 ON)

void setRed() { 
    ledcWrite(RED_LED_GPIO, 0); 
    ledcWrite(GREEN_LED_GPIO, 256);
}

void setGreen() {
    ledcWrite(RED_LED_GPIO, 256); 
    ledcWrite(GREEN_LED_GPIO, 0);
}

void setOrange() {
    ledcWrite(RED_LED_GPIO, 0); 
    ledcWrite(GREEN_LED_GPIO, 128);
}

void setYellow() {
    ledcWrite(RED_LED_GPIO, 0); 
    ledcWrite(GREEN_LED_GPIO, 0);
}

void setOff() {
    ledcWrite(RED_LED_GPIO, 256); 
    ledcWrite(GREEN_LED_GPIO, 256);
}

// void setLED_PWM(int redValue, int greenValue) {
//     ledcWrite(LED_CHANNEL_R, redValue);
//     ledcWrite(LED_CHANNEL_G, greenValue);
// }

// void setRed() { setLED_PWM(0, 255); }         
// void setGreen() { setLED_PWM(255, 0); }
// void setOrange() { setLED_PWM(RED_BRIGHTNESS, GREEN_BRIGHTNESS); }
// void setYellow() { setLED_PWM(0, 0); }
// void setOff() { setLED_PWM(255, 255); }         

// ===== Camera init =====
bool initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    if(psramFound()){
        config.frame_size = FRAMESIZE_VGA;
        config.jpeg_quality = 12;
        config.fb_count = 2;
    } else {
        config.frame_size = FRAMESIZE_QVGA;
        config.jpeg_quality = 15;
        config.fb_count = 1;
    }

    return esp_camera_init(&config) == ESP_OK;
}

// ===== WiFi =====
void connectWiFi() {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    while(WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.printf("\nWiFi connected, IP: %s\n", WiFi.localIP().toString().c_str());
}

// ===== JPEG HTTP 전송 =====
bool postJpeg(const uint8_t* buf, size_t len) {
    WiFiClient client;
    if(!client.connect(SERVER_HOST, SERVER_PORT)) {
      // HTTP 서버 연결 실패
      setYellow();
      Serial.println("[HTTP] Connection failed.");
      return false;
    }
    client.printf("POST /upload HTTP/1.1\r\n");
    client.printf("Host: %s\r\n", SERVER_HOST);
    client.printf("Content-Type: image/jpeg\r\n");
    client.printf("Content-Length: %d\r\n", len);
    client.printf("ESP-ID: %s\r\n", ESP32_ID);
    client.print("Connection: close\r\n\r\n");
    
    size_t written = client.write(buf, len);
    client.flush();
    
    if (written != len) {
        // 데이터 전송 실패
        Serial.printf("[HTTP] Incomplete data sent: %d of %d\n", written, len);
        client.stop();
        setYellow();
        return false;
    }


    String resp;
    uint32_t start = millis();
    while(millis()-start < 8000){
        while(client.available()) resp += char(client.read());
        if(!client.connected()) break;
        delay(10);
    }
    client.stop();

    if (resp.indexOf("200 OK") > 0) {
        setLEDByState(); // 성공: LED 상태를 현재 currentState(ALERT/NEARBY/CLEAR)에 맞게 복구
        Serial.println("[HTTP] Upload success (200 OK).");
        return true;
    }

    // 실패: 실패 상태(Yellow)를 표시하고, 이전 경고 상태를 복원하지 않습니다.
    Serial.printf("[HTTP] Upload failed. Response: %s\n", resp.c_str());
    setYellow(); 
    return false;
}

// ===== LED 상태 관리 =====
String currentState = "CLEAR"; // 기본 상태

void setLEDByState() {
    if(currentState == "ALERT") setRed();
    else if(currentState == "NEARBY") setOrange();
    else setGreen();
}

// ===== MQTT Callback =====
void callback(char* topic, byte* payload, unsigned int length){
    String received_topic = String(topic);

    if(received_topic.startsWith("farm/alert/")){
        currentState = "ALERT";
        setLEDByState();
        Serial.println("[MQTT] ALERT received");
    }
    else if(received_topic.startsWith("farm/nearby/")){
        currentState = "NEARBY";
        setLEDByState();
        Serial.println("[MQTT] NEARBY received");
    }
    else if(received_topic.startsWith("farm/clear/")){
        currentState = "CLEAR";
        setLEDByState();
        Serial.println("[MQTT] CLEAR received");
    }

    if(received_topic == CAPTURE_TOPIC){
        Serial.println("[Capture] Taking photo...");
        camera_fb_t* fb = esp_camera_fb_get();
        if(fb){
            if(!postJpeg(fb->buf, fb->len)){
                Serial.println("[Capture] Post failed.");
            }
            esp_camera_fb_return(fb);
        } else {
            Serial.println("[Capture] Camera capture failed!");
            setYellow();
        }
    }
}


// ===== MQTT 연결 =====
void connectMQTT(){
    while(!mqttClient.connected()){
        Serial.print("Attempting MQTT connection...");
        if(mqttClient.connect(ESP32_ID)){
            Serial.println("connected");
            
            snprintf(ALERT_TOPIC,sizeof(ALERT_TOPIC),"farm/alert/%s",ESP32_ID);
            snprintf(NEARBY_TOPIC,sizeof(NEARBY_TOPIC),"farm/nearby/%s",ESP32_ID);
            snprintf(CLEAR_TOPIC,sizeof(CLEAR_TOPIC),"farm/clear/%s",ESP32_ID);

            mqttClient.subscribe(ALERT_TOPIC);
            mqttClient.subscribe(NEARBY_TOPIC);
            mqttClient.subscribe(CLEAR_TOPIC);
            mqttClient.subscribe(CAPTURE_TOPIC);

            Serial.printf("Subscribed to: %s, %s, %s, %s\n", ALERT_TOPIC, NEARBY_TOPIC, CLEAR_TOPIC, CAPTURE_TOPIC);

        } else {
            Serial.print("failed, rc=");
            Serial.print(mqttClient.state());
            Serial.println(" retrying in 1 second");
            delay(1000);
        }
    }
}

// ===== Setup =====
void setup(){
    Serial.begin(115200);

    pinMode(FLASH_LED_GPIO,OUTPUT);
    digitalWrite(FLASH_LED_GPIO,LOW);
    
    // LED 채널 연결 (SD 핀 충돌 방지 문제에서 자유로워짐)
    ledcAttachChannel(RED_LED_GPIO, PWM_FREQ, PWM_RESOLUTION, LED_CHANNEL_R);
    ledcAttachChannel(GREEN_LED_GPIO, PWM_FREQ, PWM_RESOLUTION, LED_CHANNEL_G);

    setOff();

    connectWiFi();
    setGreen();

    mqttClient.setServer(MQTT_BROKER,MQTT_PORT);
    mqttClient.setCallback(callback);

    if(!initCamera()){ Serial.println("Camera init failed"); ESP.restart(); }
    // if(!initSD()){ Serial.println("SD init failed"); } // SD 초기화 제거
}

// ===== Loop =====
void loop(){
    if(!mqttClient.connected()) connectMQTT();
    mqttClient.loop();
    // retrySD();  // SD 재전송 제거
    delay(10);
}