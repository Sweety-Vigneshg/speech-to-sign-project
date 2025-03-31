from flask import Flask, render_template, jsonify, request, Response
import cv2
import mediapipe as mp
import numpy as np
import time
from spellchecker import SpellChecker
import pyttsx3
import threading
import base64

app = Flask(__name__)

# MediaPipe Hands configuration
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Thread safety
hands_lock = threading.Lock()
tts_lock = threading.Lock()

# Global variables
spell = SpellChecker()
detection_delay = 1.5
last_detection_time = time.time()

def detect_manual_sign(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append((landmark.x, landmark.y, landmark.z))

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

     # Detect "A" (Thumb and index finger touching)
    if distance(thumb_tip, index_tip) < 0.05:
        return 'A'

    # Detect "B" (Fingers curled into a C shape)
    if (distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) > 0.1 and
        distance(ring_tip, thumb_tip) > 0.1 and
        distance(pinky_tip, thumb_tip) > 0.1):
        return 'B'
    
    # Detect "C" (All fingers extended)
    fingertips = [index_tip, middle_tip, ring_tip, pinky_tip]
    y_coords = [tip[1] for tip in fingertips]
    if np.std(y_coords) < 0.02:
        return 'C'

    # Detect "D" (Index finger extended, others curled)
    if (distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) < 0.05 and
        distance(ring_tip, thumb_tip) < 0.05 and
        distance(pinky_tip, thumb_tip) < 0.05):
        return 'D'

    # Detect "E" (All fingers curled into a fist)
    if (distance(index_tip, thumb_tip) < 0.05 and
        distance(middle_tip, thumb_tip) < 0.05 and
        distance(ring_tip, thumb_tip) < 0.05 and
        distance(pinky_tip, thumb_tip) < 0.05):
        return 'E'

    # Detect "F" (Thumb and index finger touching, others extended)
    if (distance(thumb_tip, index_tip) < 0.05 and
        distance(middle_tip, thumb_tip) > 0.1 and
        distance(ring_tip, thumb_tip) > 0.1 and
        distance(pinky_tip, thumb_tip) > 0.1):
        return 'F'

    # Detect "G" (Index finger pointing, thumb touching middle finger)
    if (distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) < 0.05):
        return 'G'

    # Detect "H" (Index and middle fingers extended, others curled)
    if (distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) > 0.1 and
        distance(ring_tip, thumb_tip) < 0.05 and
        distance(pinky_tip, thumb_tip) < 0.05):
        return 'H'

    # Detect "I" (Pinky finger extended, others curled)
    if (distance(pinky_tip, thumb_tip) > 0.1 and
        distance(index_tip, thumb_tip) < 0.05 and
        distance(middle_tip, thumb_tip) < 0.05 and
        distance(ring_tip, thumb_tip) < 0.05):
        return 'I'

    # Detect "J" (Pinky finger extended with a hook, others curled)
    if (distance(pinky_tip, thumb_tip) > 0.1 and
        pinky_tip[1] < thumb_tip[1]):  # Pinky above thumb
        return 'J'

    # Detect "K" (Index and middle fingers extended, thumb touching ring finger)
    if (distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) > 0.1 and
        distance(ring_tip, thumb_tip) < 0.05):
        return 'K'

    # Detect "L" (Index finger and thumb extended, others curled)
    if (distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) < 0.05 and
        distance(ring_tip, thumb_tip) < 0.05 and
        distance(pinky_tip, thumb_tip) < 0.05):
        return 'L'

    # Detect "M" (All fingers curled, thumb over fingers)
    if (distance(index_tip, thumb_tip) < 0.05 and
        distance(middle_tip, thumb_tip) < 0.05 and
        distance(ring_tip, thumb_tip) < 0.05 and
        distance(pinky_tip, thumb_tip) < 0.05 and
        thumb_tip[1] > index_tip[1]):  # Thumb above index
        return 'M'

    # Detect "N" (Index and middle fingers curled, thumb over fingers)
    if (distance(index_tip, thumb_tip) < 0.05 and
        distance(middle_tip, thumb_tip) < 0.05 and
        distance(ring_tip, thumb_tip) > 0.1 and
        distance(pinky_tip, thumb_tip) > 0.1):
        return 'N'

    # Detect "O" (Fingers curled into an O shape)
    if (distance(index_tip, thumb_tip) < 0.05 and
        distance(middle_tip, thumb_tip) < 0.05 and
        distance(ring_tip, thumb_tip) < 0.05 and
        distance(pinky_tip, thumb_tip) < 0.05 and
        thumb_tip[0] > index_tip[0]):  # Thumb to the right of index
        return 'O'

    # Detect "P" (Index finger pointing down, thumb extended)
    if (distance(index_tip, thumb_tip) > 0.1 and
        index_tip[1] > thumb_tip[1]):  # Index below thumb
        return 'P'

    # Detect "Q" (Index finger pointing, thumb touching middle finger)
    if (distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) < 0.05):
        return 'Q'

    # Detect "R" (Index and middle fingers crossed)
    if (distance(index_tip, middle_tip) < 0.05):
        return 'R'

    # Detect "S" (All fingers curled into a fist)
    if (distance(index_tip, thumb_tip) < 0.05 and
        distance(middle_tip, thumb_tip) < 0.05 and
        distance(ring_tip, thumb_tip) < 0.05 and
        distance(pinky_tip, thumb_tip) < 0.05):
        return 'S'

    # Detect "T" (Thumb between index and middle fingers)
    if (distance(thumb_tip, index_tip) < 0.05 and
        distance(thumb_tip, middle_tip) < 0.05):
        return 'T'

    # Detect "U" (Index and middle fingers extended, others curled)
    if (distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) > 0.1 and
        distance(ring_tip, thumb_tip) < 0.05 and
        distance(pinky_tip, thumb_tip) < 0.05):
        return 'U'

    # Detect "V" (Index and middle fingers extended and apart)
    if (distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) > 0.1 and
        distance(index_tip, middle_tip) > 0.1):
        return 'V'

    # Detect "W" (Index, middle, and ring fingers extended)
    if (distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) > 0.1 and
        distance(ring_tip, thumb_tip) > 0.1 and
        distance(pinky_tip, thumb_tip) < 0.05):
        return 'W'

    # Detect "X" (Index finger curled, others extended)
    if (distance(index_tip, thumb_tip) < 0.05 and
        distance(middle_tip, thumb_tip) > 0.1 and
        distance(ring_tip, thumb_tip) > 0.1 and
        distance(pinky_tip, thumb_tip) > 0.1):
        return 'X'

    # Detect "Y" (Thumb and pinky extended, others curled)
    if (distance(thumb_tip, pinky_tip) > 0.1 and
        distance(index_tip, thumb_tip) < 0.05 and
        distance(middle_tip, thumb_tip) < 0.05 and
        distance(ring_tip, thumb_tip) < 0.05):
        return 'Y'

    # Detect "Z" (Index finger pointing, thumb touching ring finger)
    if (distance(index_tip, thumb_tip) > 0.1 and
        distance(ring_tip, thumb_tip) < 0.05):
        return 'Z'

    # Detect Numbers (0-9)
    # Detect "0" (Fingers curled into a circle)
    if (distance(index_tip, thumb_tip) < 0.05 and
        distance(middle_tip, thumb_tip) < 0.05 and
        distance(ring_tip, thumb_tip) < 0.05 and
        distance(pinky_tip, thumb_tip) < 0.05):
        return '0'

    # Detect "1" (Index finger extended, others curled)
    if (distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) < 0.05 and
        distance(ring_tip, thumb_tip) < 0.05 and
        distance(pinky_tip, thumb_tip) < 0.05):
        return '1'

    # Detect "2" (Index and middle fingers extended, others curled)
    if (distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) > 0.1 and
        distance(ring_tip, thumb_tip) < 0.05 and
        distance(pinky_tip, thumb_tip) < 0.05):
        return '2'

    # Detect "3" (Index, middle, and ring fingers extended)
    if (distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) > 0.1 and
        distance(ring_tip, thumb_tip) > 0.1 and
        distance(pinky_tip, thumb_tip) < 0.05):
        return '3'

    # Detect "4" (All fingers extended)
    if (distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) > 0.1 and
        distance(ring_tip, thumb_tip) > 0.1 and
        distance(pinky_tip, thumb_tip) > 0.1):
        return '4'

    # Detect "5" (All fingers extended and spread apart)
    if (distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) > 0.1 and
        distance(ring_tip, thumb_tip) > 0.1 and
        distance(pinky_tip, thumb_tip) > 0.1 and
        distance(index_tip, middle_tip) > 0.1 and
        distance(middle_tip, ring_tip) > 0.1 and
        distance(ring_tip, pinky_tip) > 0.1):
        return '5'

    # Detect "6" (Thumb touching pinky, others extended)
    if (distance(thumb_tip, pinky_tip) < 0.05 and
        distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) > 0.1 and
        distance(ring_tip, thumb_tip) > 0.1):
        return '6'

    # Detect "7" (Thumb touching ring finger, others extended)
    if (distance(thumb_tip, ring_tip) < 0.05 and
        distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) > 0.1 and
        distance(pinky_tip, thumb_tip) > 0.1):
        return '7'

    # Detect "8" (Thumb touching middle finger, others extended)
    if (distance(thumb_tip, middle_tip) < 0.05 and
        distance(index_tip, thumb_tip) > 0.1 and
        distance(ring_tip, thumb_tip) > 0.1 and
        distance(pinky_tip, thumb_tip) > 0.1):
        return '8'

    # Detect "9" (Thumb touching index finger, others extended)
    if (distance(thumb_tip, index_tip) < 0.05 and
        distance(middle_tip, thumb_tip) > 0.1 and
        distance(ring_tip, thumb_tip) > 0.1 and
        distance(pinky_tip, thumb_tip) > 0.1):
        return '9'

    # Detect space (All fingers extended and spread apart)
    if (distance(index_tip, thumb_tip) > 0.1 and
        distance(middle_tip, thumb_tip) > 0.1 and
        distance(ring_tip, thumb_tip) > 0.1 and
        distance(pinky_tip, thumb_tip) > 0.1 and
        distance(index_tip, middle_tip) > 0.1 and
        distance(middle_tip, ring_tip) > 0.1 and
        distance(ring_tip, pinky_tip) > 0.1):
        return ' '

    return None  # No manual sign detected


@app.route('/')
def home():
    return render_template('1.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global last_detection_time
    
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data'}), 400
            
        header, encoded = data['image'].split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        with hands_lock:
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        detected_text = ""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                sign = detect_manual_sign(hand_landmarks)
                if sign:
                    current_time = time.time()
                    if current_time - last_detection_time >= detection_delay:
                        detected_text = sign
                        last_detection_time = current_time
                    break

        return jsonify({'text': detected_text})

    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/speak', methods=['POST'])
def speak_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        with tts_lock:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()

        return jsonify({'status': 'success'})

    except Exception as e:
        print(f"TTS error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)