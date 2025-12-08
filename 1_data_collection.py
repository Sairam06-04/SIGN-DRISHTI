import cv2 # Changed from cv
import mediapipe as mp
import numpy as np
import os
import csv

# Initialize MediaPipe Hands (can detect up to 2 hands)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# --- Configuration ---
DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

DATA_FILE = os.path.join(DATA_DIR, 'sign_language_data.csv')

# How many samples (frames) to capture per sign
NUM_SAMPLES = 100 # Adjust as needed

# --- Prepare CSV ---
# Header: label + 126 features (21 landmarks * 3 coords * 2 hands)
header = ['label']
# Left Hand Landmarks (if detected, otherwise zeros)
for i in range(21): header.extend([f'lh_{i}_x', f'lh_{i}_y', f'lh_{i}_z'])
# Right Hand Landmarks (if detected, otherwise zeros)
for i in range(21): header.extend([f'rh_{i}_x', f'rh_{i}_y', f'rh_{i}_z'])

# Check if file exists to append or write header
write_header = not os.path.exists(DATA_FILE)
if write_header:
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    print("Created data file and wrote header.")
else:
    print("Data file exists. Will append new data.")

# --- Get Label ---
label_name = input("Enter the label name for this data collection (e.g., hello, thanks, yes): ").strip().lower()
if not label_name:
    print("Label name cannot be empty. Exiting.")
    exit()

# --- Webcam Setup ---
cap = cv2.VideoCapture(0) # Changed from cv
print(f"\nPrepare to show the sign for: '{label_name}'.")
print(f"Press 'S' to start the capture session for {NUM_SAMPLES} samples.")

sample_counter = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1) # Changed from cv

    # Show instructions
    if sample_counter == 0:
        cv2.putText(frame, "Press 'S' to start capturing...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # Changed from cv
    else:
         cv2.putText(frame, f"Collected {sample_counter}/{NUM_SAMPLES}. Press 'Q' to exit.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # Changed from cv

    cv2.imshow("Data Collection", frame) # Changed from cv
    key = cv2.waitKey(1) & 0xFF # Changed from cv

    if key == ord('s') and sample_counter < NUM_SAMPLES:
        print("Starting manual capture...")

        while sample_counter < NUM_SAMPLES:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1) # Changed from cv
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Changed from cv
            results = hands.process(rgb_frame)

            # Draw landmarks if hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show capture instructions and count
            cv2.putText(frame, f"Press 'C' to capture ({sample_counter}/{NUM_SAMPLES})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # Changed from cv
            cv2.putText(frame, f"Label: {label_name}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) # Changed from cv
            cv2.putText(frame, "Press 'Q' to quit early.", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) # Changed from cv
            cv2.imshow("Data Collection", frame) # Changed from cv

            capture_key = cv2.waitKey(1) & 0xFF # Changed from cv

            if capture_key == ord('c'):
                # Extract landmarks
                left_hand_landmarks = np.zeros(21 * 3)
                right_hand_landmarks = np.zeros(21 * 3)

                if results.multi_hand_landmarks:
                    for idx, hand_info in enumerate(results.multi_handedness):
                        hand_landmarks = results.multi_hand_landmarks[idx]
                        # Normalize landmarks relative to the wrist (landmark 0)
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

                # Combine landmarks into one row
                row = [label_name] + list(left_hand_landmarks) + list(right_hand_landmarks)

                # Append to CSV
                with open(DATA_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                print(f"Captured sample {sample_counter+1}/{NUM_SAMPLES} for '{label_name}'")
                sample_counter += 1

            elif capture_key == ord('q'):
                print("Quitting capture early.")
                break # Exit the inner capture loop

        if sample_counter >= NUM_SAMPLES:
            print(f"Target number of samples ({NUM_SAMPLES}) captured for '{label_name}'!")
        # Fall back to the outer loop after finishing/quitting capture

    elif key == ord('q'):
        break # Exit the outer loop

cap.release()
cv2.destroyAllWindows() # Changed from cv
hands.close()