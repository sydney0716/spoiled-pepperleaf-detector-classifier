import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

def load_models():
    """
    Load YOLOv8 detector and ResNet-18 classifier.

    Returns:
        A tuple containing:
            - YOLO interpreter and its input/output details
            - Resnet interpreter and its input/output details
            - Label list
    """
    detect_interpreter = tflite.Interpreter(model_path="/home/hoyoungchung/Desktop/yolo_float16.tflite")
    detect_interpreter.allocate_tensors()
    detect_input_details = detect_interpreter.get_input_details()
    detect_output_details = detect_interpreter.get_output_details()

    class_interpreter = tflite.Interpreter(model_path="/home/hoyoungchung/Desktop/NP-converted-resnet18.tflite")
    class_interpreter.allocate_tensors()
    class_input_details = class_interpreter.get_input_details()
    class_output_details = class_interpreter.get_output_details()

    CLASS_LABELS = ['normal', 'spoiled']

    return (
        detect_interpreter, 
        class_interpreter, 
        detect_input_details, 
        detect_output_details, 
        class_input_details, 
        class_output_details, 
        CLASS_LABELS
    )

def detect_and_classify_image(frame, model_data, conf_threshold=0.5):
    """
    Perform object detection using YOLOv8 TFLite, followed by
    classification of detected regions using Resnet-18.
    """

    if frame is None:
        print("[ERROR] Received empty frame.")
        return None, None

    (
        detect_interpreter,
        class_interpreter,
        detect_input_details,
        detect_output_details,
        class_input_details,
        class_output_details,
        CLASS_LABELS
    ) = model_data

    orig_h, orig_w = frame.shape[:2]
    result_frame = frame.copy()

    # YOLO Preprocessing
    input_height = detect_input_details[0]['shape'][1]
    input_width = detect_input_details[0]['shape'][2]

    img = cv2.resize(frame, (input_width, input_height))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    detect_interpreter.set_tensor(detect_input_details[0]['index'], img)
    detect_interpreter.invoke()

    detections = detect_interpreter.get_tensor(detect_output_details[0]['index'])[0]

    # Normalize YOLO output format
    if detections.shape[0] == 5:
        detections = detections.transpose()
    elif detections.shape[-1] != 5:
        print("[WARN] Unexpected detection shape:", detections.shape)
        return 0, frame
    
    has_spoiled = False

    # Postprocess
    for det in detections:
        x, y, w, h, conf = det

        if conf < conf_threshold:
            continue

        xmin = int((x - w/2) * orig_w)
        ymin = int((y - h/2) * orig_h)
        xmax = int((x + w/2) * orig_w)
        ymax = int((y + h/2) * orig_h)

        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(orig_w, xmax), min(orig_h, ymax)

        crop = frame[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue

        # ResNet Classification
        crop_resized = cv2.resize(crop, (224, 224))
        crop_input = np.expand_dims(crop_resized, axis=0).astype(np.float32) / 255.0

        class_interpreter.set_tensor(class_input_details[0]['index'], crop_input)
        class_interpreter.invoke()
        class_output = class_interpreter.get_tensor(class_output_details[0]['index'])
        class_id = np.argmax(class_output)

        label = CLASS_LABELS[class_id]
        color = (0, 0, 255) if label == 'spoiled' else (0, 255, 0)

        cv2.rectangle(result_frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            result_frame,
            f"{label} (conf {conf:.2f})",
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
        
        if label == 'spoiled':
            has_spoiled = True

    classification_result = 1 if has_spoiled else 0
    return classification_result, result_frame