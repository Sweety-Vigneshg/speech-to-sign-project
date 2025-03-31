from flask import Flask, render_template, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import base64
import time
from spellchecker import SpellChecker
import pyttsx3
import threading

app = Flask(__name__)

# MediaPipe Hands configuration
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Thread safety locks
hands_lock = threading.Lock()
tts_lock = threading.Lock()

# Initialize text-to-speech engine
spell = SpellChecker()

def detect_manual_sign(hand_landmarks):
    def distance(p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    try:
        thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        if distance(thumb, index) < 0.1:
            return 'A'
        elif all(distance(thumb, f) > 0.15 for f in [index, middle, ring, pinky]):
            return 'B'
        elif distance(index, middle) < 0.1 and distance(middle, ring) < 0.1:
            return 'C'
        elif distance(thumb, middle) < 0.1 and distance(index, thumb) > 0.2:
            return 'D'
        elif all(distance(thumb, f) < 0.1 for f in [index, middle, ring, pinky]):
            return 'E'
        elif distance(thumb, pinky) < 0.1 and distance(index, middle) > 0.2:
            return 'F'
        elif distance(index, middle) > 0.2 and distance(thumb, ring) < 0.1:
            return 'G'
        elif distance(index, middle) < 0.1 and distance(ring, pinky) > 0.2:
            return 'H'
        elif distance(pinky, thumb) > 0.2 and all(distance(thumb, f) < 0.1 for f in [index, middle, ring]):
            return 'I'
        elif distance(thumb, index) > 0.2 and distance(index, middle) < 0.1:
            return 'K'
        elif all(distance(thumb, f) > 0.2 for f in [middle, ring, pinky]) and distance(index, thumb) < 0.1:
            return 'L'
        elif distance(thumb, pinky) > 0.2 and distance(index, middle) < 0.1:
            return 'V'
        elif all(distance(thumb, f) > 0.2 for f in [index, middle, ring, pinky]):
            return ' '
        return ''
    except Exception as e:
        print(f"Detection error: {e}")
        return ''

@app.route('/')
def h1():
    return render_template('1.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data received'}), 400

        header, encoded = data['image'].split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400

        with hands_lock:
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        detected_text = ''
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                sign = detect_manual_sign(hand_landmarks)
                if sign:
                    detected_text = sign
                    break

        return jsonify({'text': detected_text})

    except Exception as e:
        print(f"Processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/speak', methods=['POST'])
def speak_text():
    try:
        data = request.get_json()
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text'}), 400

        with tts_lock:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()

        return jsonify({'status': 'success'})

    except Exception as e:
        print(f"TTS error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)