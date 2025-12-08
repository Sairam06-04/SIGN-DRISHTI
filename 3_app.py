import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from collections import deque

# --- Load Model ---
MODEL_FILE = os.path.join('model', 'sign_language_model.p')
if not os.path.exists(MODEL_FILE):
    print("Error: Model not found. Please run '2_train_model.py' first.")
    exit()

with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)
print("Model loaded successfully.")

# --- Initialize MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- Logic Variables ---
prediction_buffer = deque(maxlen=15) # Buffer for stabilizing predictions
STABLE_THRESHOLD = 10                # How many identical predictions needed
CONFIDENCE_THRESHOLD = 0.2          # Minimum confidence (keep it low for now)
last_stable_prediction = None
current_sentence = []
# REMOVED frame_count and PREDICT_EVERY_N_FRAMES

print("Starting real-time detection... Press 'C' to clear sentence. Press 'Q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    display_prediction = "" # Prediction to display on the hand this frame
    stable_prediction = None # Reset stable prediction each frame

    # --- Process landmarks and predict EVERY frame ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Initialize landmark arrays with zeros
    left_hand_landmarks = np.zeros(21 * 3)
    right_hand_landmarks = np.zeros(21 * 3)

    if results.multi_hand_landmarks:
        for idx, hand_info in enumerate(results.multi_handedness):
            hand_landmarks = results.multi_hand_landmarks[idx]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Draw landmarks

            # Normalize landmarks relative to the wrist
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])

            hand_type = hand_info.classification[0].label
            if hand_type == 'Left':
                left_hand_landmarks = np.array(landmarks)
            elif hand_type == 'Right':
                right_hand_landmarks = np.array(landmarks)

        # Combine landmarks for prediction
        combined_landmarks = np.concatenate([left_hand_landmarks, right_hand_landmarks]).reshape(1, -1)
        combined_landmarks = np.nan_to_num(combined_landmarks) # Replace potential NaNs

        # --- Make Prediction ---
        try:
            prediction_proba = model.predict_proba(combined_landmarks)[0]
            confidence = np.max(prediction_proba)
            predicted_class_index = np.argmax(prediction_proba)

            if confidence >= CONFIDENCE_THRESHOLD:
                predicted_label = model.classes_[predicted_class_index]
                display_prediction = predicted_label # Store for display
                prediction_buffer.append(predicted_label)

                # --- Check for Stability ---
                if len(prediction_buffer) == prediction_buffer.maxlen:
                    if all(p == predicted_label for p in list(prediction_buffer)[-STABLE_THRESHOLD:]):
                        stable_prediction = predicted_label
            else:
                prediction_buffer.clear() # Clear buffer if confidence is low

        except Exception as e:
            # print(f"Error during prediction: {e}") # Uncomment to see errors
            prediction_buffer.clear()
    else:
        # No hands detected, clear buffer
        prediction_buffer.clear()
        last_stable_prediction = None # Reset last prediction if hands disappear

    # --- Sentence Building Logic ---
    if stable_prediction is not None and stable_prediction != last_stable_prediction:
        current_sentence.append(stable_prediction)
        current_sentence.append(" ") # Add space
        last_stable_prediction = stable_prediction

    # --- Display Current Prediction (near top corner) ---
    cv2.putText(frame, display_prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # --- Display Sentence at Bottom ---
    bar_height = 60
    cv2.rectangle(frame, (0, frame.shape[0] - bar_height), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    display_text = "".join(current_sentence)
    cv2.putText(frame, display_text, (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.putText(frame, "Press 'C' to clear | 'Q' to quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Sign Language Detector", frame)

    # --- Key Press Logic ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        current_sentence = []
        last_stable_prediction = None
        prediction_buffer.clear()

cap.release()
cv2.destroyAllWindows()
hands.close()