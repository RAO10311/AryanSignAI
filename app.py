from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import threading
from datetime import datetime

app = Flask(__name__)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.65,
                       min_tracking_confidence=0.60)
drawing = mp.solutions.drawing_utils

status = {"found": False, "fingers": 0, "label": "Starting camera...",
          "confidence": 0, "time": "", "camera_ok": False,
          "error": "Camera has not started"}
camera_lock = threading.Lock()
camera = None

def open_camera():
    # Try common Windows webcam backends and camera indexes.
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for index in (0, 1, 2):
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap
            cap.release()
    return None

def finger_count(hand, side):
    p = hand.landmark
    count = sum(p[t].y < p[k].y for t, k in [(8,6),(12,10),(16,14),(20,18)])
    thumb = p[4].x < p[3].x if side == "Right" else p[4].x > p[3].x
    return int(count) + int(thumb)

def label_for(n):
    return ["Closed hand / fist-like pose", "One finger extended",
            "Two fingers extended", "Three fingers extended",
            "Four fingers extended", "Open palm"][n]

def placeholder_frame(message):
    frame = 255 * (cv2.UMat(480, 800, cv2.CV_8UC3).get() * 0)
    cv2.putText(frame, message, (35, 220), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (40, 40, 220), 2, cv2.LINE_AA)
    cv2.putText(frame, "Check Windows camera privacy and close other camera apps",
                (35, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (70, 70, 70), 1, cv2.LINE_AA)
    return frame

def frames():
    global camera
    with camera_lock:
        camera = open_camera()
        if camera is None:
            status.update(found=False, camera_ok=False,
                          label="Camera not available", confidence=0,
                          time=datetime.now().strftime("%H:%M:%S"),
                          error="Could not open webcam. Check Windows camera permission, close other apps, and try again.")
            frame = placeholder_frame("WEBCAM NOT FOUND")
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            return

        status.update(camera_ok=True, error="", label="Show one hand to the camera")
        try:
            while True:
                ok, frame = camera.read()
                if not ok or frame is None:
                    status.update(found=False, camera_ok=False,
                                  label="Camera frame unavailable",
                                  error="Camera opened but no frames arrived. Reconnect the webcam.")
                    break
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    side, confidence = "Right", 0
                    if result.multi_handedness:
                        cls = result.multi_handedness[0].classification[0]
                        side, confidence = cls.label, round(cls.score * 100)
                    n = finger_count(hand, side)
                    label = label_for(n)
                    drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                    status.update(found=True, fingers=n, label=label,
                                  confidence=confidence,
                                  time=datetime.now().strftime("%H:%M:%S"))
                else:
                    status.update(found=False, fingers=0, label="No hand detected",
                                  confidence=0,
                                  time=datetime.now().strftime("%H:%M:%S"))
                cv2.putText(frame, status["label"], (18, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (80, 235, 180), 2)
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        finally:
            if camera is not None:
                camera.release()
                camera = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video_feed")
def video():
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

@app.route("/status")
def get_status():
    return jsonify(status)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
