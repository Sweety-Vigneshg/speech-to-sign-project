from flask import Flask, render_template,  send_from_directory, request, jsonify, Response
import os
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3

app = Flask(__name__)


# MODEL 1 CODE

VIDEO_FOLDER = os.path.join("static", "signs")
app.config["VIDEO_FOLDER"] = VIDEO_FOLDER

# Alphabet fallback (A-Z sign videos)
ALPHABET_MAP = {chr(i): f"{chr(i).upper()}.mp4" for i in range(97, 123)}

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    sentence = data.get("sentence", "").strip().lower()
    
    if not sentence:
        return jsonify({"videos": ["/static/signs/Talk.mp4"]})

    video_files = []
    
    for word in sentence.split():
        if word in VIDEO_MAP:
            video_files.append(f"/static/signs/{VIDEO_MAP[word]}")
        else:
            video_files.extend(f"/static/signs/{ALPHABET_MAP[char]}" for char in word if char in ALPHABET_MAP)

    if not video_files:
        video_files.append("/static/signs/Talk.mp4")

    return jsonify({"videos": video_files})


# MODEL 11 CODE

VIDEO_DIR = "static/signs2"

def get_video_files(word):
    videos = []
    for char in word.upper():
        filename = f"{char}.mp4"
        filepath = os.path.join(VIDEO_DIR, filename)
        if os.path.exists(filepath):
            videos.append(filename)
    return videos

@app.route('/translate')
def translate1():
    word = request.args.get('word', '').lower() 
    videos = get_video_files(word)  
    return jsonify({"videos": videos})

@app.route('/static/signs2/<path:filename>')
def serve_video(filename):
    return send_from_directory(VIDEO_DIR, filename)


# MODEL 2 CODE

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from spellchecker import SpellChecker  # For autocorrection
import pyttsx3  # For text-to-speech
import time  # For time-based delay

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("sign_model.h5")

# Load class labels (A-Z)
labels = {i: chr(65 + i) for i in range(26)}  # Assuming A-Z (26 classes)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                       min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Global variables for text and detection state
detected_text = ""
full_text = ""
is_running = True
last_detection_time = time.time()  # Track the last detection time
detection_delay = 1.5  # Delay in seconds between detections

# Initialize spell checker
spell = SpellChecker()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to detect manual hand signs
def detect_manual_sign(hand_landmarks):
    # Extract landmark positions
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append((landmark.x, landmark.y, landmark.z))

    # Define conditions for each letter and number
    thumb_tip = landmarks[4]  # Thumb tip
    index_tip = landmarks[8]  # Index finger tip
    middle_tip = landmarks[12]  # Middle finger tip
    ring_tip = landmarks[16]  # Ring finger tip
    pinky_tip = landmarks[20]  # Pinky finger tip

     # Calculate distances between fingertips
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

# Function to generate video feed
def generate_frames():
    global detected_text, full_text, is_running, last_detection_time
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if is_running:
            # Flip image horizontally for natural view
            frame = cv2.flip(frame, 1)

            # Convert frame to RGB (for Mediapipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Detect manual sign
                    predicted_sign = detect_manual_sign(hand_landmarks)

                    if predicted_sign:
                        current_time = time.time()
                        # Append the sign only if the delay has passed
                        if current_time - last_detection_time >= detection_delay:
                            detected_text = predicted_sign
                            full_text += predicted_sign  # Append to full text
                            last_detection_time = current_time  # Update last detection time

                    # Draw Hand Landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display Prediction
            cv2.putText(frame, f"Sign: {detected_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Full Text: {full_text}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to detect sign
@app.route('/detect_sign')
def detect_sign():
    global full_text
    corrected_text = " ".join([spell.correction(word) for word in full_text.split()])  # Autocorrect the full text
    return jsonify({'text': corrected_text})

# Route to stop detection
@app.route('/stop')
def stop_detection():
    global is_running
    is_running = False
    return jsonify({'status': 'stopped'})

# Route to restart detection
@app.route('/restart')
def restart_detection():
    global is_running, detected_text, full_text, last_detection_time
    is_running = True
    detected_text = ""
    full_text = ""
    last_detection_time = time.time()  # Reset the last detection time
    
    return jsonify({'status': 'restarted'})

# Route to speak text
@app.route('/speak')
def speak_text():
    global full_text
    corrected_text = " ".join([spell.correction(word) for word in full_text.split()])  # Autocorrect the full text
    engine.say(corrected_text)  # Speak the corrected text
    engine.runAndWait()
    return jsonify({'status': 'spoken'})

# Routes HTML Pages
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/model1')
def model1():
    return render_template("model1.html")

@app.route('/model11')
def model11():
    return render_template("model11.html")

@app.route('/model2')
def model2():
    return render_template("model2.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/converter')
def converter():
    return render_template("converter.html")

if __name__ == '__main__':
    app.run(debug=True)
