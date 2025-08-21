import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import speech_recognition as sr
import threading

# -------------------------------
# 1. TTS Engine Setup
# -------------------------------
tts_engine = pyttsx3.init()

def speak_objects(objects):
    if objects:
        speech_text = "Detected " + " and ".join(objects)
        # Run TTS in a separate thread so it doesn't block OpenCV
        def tts_thread():
            tts_engine.say(speech_text)
            tts_engine.runAndWait()
        threading.Thread(target=tts_thread).start()

# -------------------------------
# 2. Distance Estimation Setup
# -------------------------------
KNOWN_WIDTHS = {
    "person": 50,
    "bottle": 7,
    "car": 180,
    "dog": 30,
    "cat": 20
}
FOCAL_LENGTH_PX = 700

def estimate_distance(known_width_cm, focal_length_px, pixel_width):
    if pixel_width == 0:
        return None
    return (known_width_cm * focal_length_px) / pixel_width

# -------------------------------
# 3. YOLOv8 Model Setup
# -------------------------------
model = YOLO('yolov8n.pt')

# -------------------------------
# 4. Speech Recognition Setup
# -------------------------------
recognizer = sr.Recognizer()

def listen_for_command(timeout=3):
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=3)
            command = recognizer.recognize_google(audio).lower()
            return command
        except (sr.UnknownValueError, sr.WaitTimeoutError):
            return ""
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return ""

# -------------------------------
# 5. Capture Single Frame & Detect
# -------------------------------
def capture_and_detect():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to capture image")
        return
    
    results = model(frame)[0]
    detected_objects = []
    
    for box, conf, cls_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        label = model.names[int(cls_id.cpu().item())]
        pixel_width = x2 - x1
        
        distance_cm = estimate_distance(KNOWN_WIDTHS.get(label, 20), FOCAL_LENGTH_PX, pixel_width)
        text = f"{label}: {distance_cm:.1f} cm" if distance_cm else label
        detected_objects.append(text)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        y_text = y1 - 15 if y1 - 15 > 15 else y1 + 15
        cv2.putText(frame, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    # Speak objects **before showing image**
    speak_objects(detected_objects)
    
    # Show image
    cv2.imshow("Captured Image Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------------
# 6. Main Loop: Wait for 'scan'
# -------------------------------
if __name__ == "__main__":
    print("===== YOLOv8 Single Capture Detection =====")
    while True:
        print("Say 'scan' to capture an image and detect objects...")
        command = listen_for_command(timeout=5)
        if "scan" in command:
            print("Capturing image and detecting objects...")
            capture_and_detect()