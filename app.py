from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
from datetime import datetime

app = Flask(__name__)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.65,
                       min_tracking_confidence=0.60)
drawing = mp.solutions.drawing_utils
status = {"found": False, "fingers": 0, "label": "Show one hand to the camera",
          "confidence": 0, "time": ""}
camera = None

def finger_count(hand, side):
    p = hand.landmark
    count = sum(p[t].y < p[k].y for t, k in [(8,6),(12,10),(16,14),(20,18)])
    thumb = p[4].x < p[3].x if side == "Right" else p[4].x > p[3].x
    return int(count) + int(thumb)

def label_for(n):
    return ["Closed hand / fist-like pose", "One finger extended",
            "Two fingers extended", "Three fingers extended",
            "Four fingers extended", "Open palm"][n]

def frames():
    global camera
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    while camera.isOpened():
        ok, frame = camera.read()
        if not ok:
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
                          confidence=confidence, time=datetime.now().strftime("%H:%M:%S"))
        else:
            status.update(found=False, fingers=0, label="No hand detected",
                          confidence=0, time=datetime.now().strftime("%H:%M:%S"))
        cv2.putText(frame, status["label"], (18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    .7, (80, 235, 180), 2)
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
    camera.release()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video_feed")
def video():
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def get_status():
    return jsonify(status)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
