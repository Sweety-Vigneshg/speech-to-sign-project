from flask import request
import cv2
import numpy as np

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    file = request.files['frame']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Process image with MediaPipe
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    detected_text = ""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            detected_text = detect_manual_sign(hand_landmarks)
    
    return jsonify({'text': detected_text})