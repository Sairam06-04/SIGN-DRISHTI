import sys
import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque
import time

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
    QPushButton, QTextEdit, QComboBox, QCompleter, QLineEdit, QSizePolicy
)
from PySide6.QtCore import (
    Qt, QThread, Signal, Slot, QTimer
)
from PySide6.QtGui import QImage, QPixmap, QFont

# --- EXTERNAL LIBRARY IMPORTS ---
try:
    from googletrans import Translator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    print("Warning: 'googletrans' library not found. Translation feature will be disabled.")
    TRANSLATOR_AVAILABLE = False
    
try:
    from gtts import gTTS
    import pygame 
    TTS_AVAILABLE = True
except ImportError:
    print("Warning: 'gTTS' or 'pygame' libraries not found. Text-to-Voice feature will be disabled.")
    TTS_AVAILABLE = False
# --------------------------------------------------------

# --- CONFIGURATION & UI/UX DEFINITIONS (DYNAMIC WIDTH) ---

# --- GENERAL CONFIG ---
MODEL_FILE = os.path.join('model', 'sign_language_model.p')
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
STABLE_THRESHOLD = 3  
CONFIDENCE_THRESHOLD = 0.2
TTS_TEMP_FILE = "temp_audio.mp3" 

# --- SIZING CONSTANTS (MODIFY THESE TO ADJUST PROPORTIONS) ---
# NOTE: Width is dynamic, Height is fixed.

# Heights
CAMERA_FIXED_HEIGHT = 480     # Height of the camera display area
SENTENCE_BOX_HEIGHT = 90      # Height of the Recognized Sentence text box 
BUTTON_HEIGHT = 55            # Height of the main buttons (START, CLEAR, QUIT)
CONTROL_BUTTON_HEIGHT = 36    # Height of buttons in the control panel (Convert/TTS/ComboBox)

# Minimum Widths (Set minimum floor for dynamic elements)
CAMERA_MIN_WIDTH = 500        
LANGUAGE_BOX_MIN_WIDTH = 200   #250 


# --- COLOR & FONT DEFINITIONS ---
BG_COLOR_PRIMARY = "#1A1616"
BG_COLOR_SECONDARY = "#332E2E"
BG_COLOR_BUTTON = "#686060"
BG_COLOR_ACCENT_RED = "#750C0C"
BG_COLOR_ACCENT_GREEN = "#3BB854"
BOX_CORNER_RADIUS_LARGE = "25px"
BOX_CORNER_RADIUS_MEDIUM = "15px"
BOX_CORNER_RADIUS_SMALL = "10px"
FONT_NAME = "Roboto"
FONT_SIZE_BODY = "16pt"
FONT_SIZE_SMALL = "12pt" 


# --- 1. Worker Thread Class (No changes) ---
class VideoWorker(QThread):
    frame_ready = Signal(QImage)
    word_ready = Signal(str)     

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.running = True
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.prediction_buffer = deque(maxlen=20) 
        self.last_stable_prediction = None
        self.STABLE_THRESHOLD = STABLE_THRESHOLD
        self.CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        if not cap.isOpened():
            print("Worker: ERROR - Cannot open webcam.")
            self.running = False
            return

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Worker: Frame read failed.")
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.flip(frame, 1)

            processed_frame, stable_prediction = self.process_frame(frame)
            
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            self.frame_ready.emit(image) 
            
            if stable_prediction and stable_prediction != self.last_stable_prediction:
                self.word_ready.emit(stable_prediction) 
                self.last_stable_prediction = stable_prediction
            
            self.msleep(1)
        
        cap.release()
        self.hands.close()
        print("Worker: Camera released.")

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        left_hand_landmarks, right_hand_landmarks = np.zeros(63), np.zeros(63)
        stable_prediction = None
        display_prediction = ""

        if results.multi_hand_landmarks:
            for idx, hand_info in enumerate(results.multi_handedness):
                hand_landmarks = results.multi_hand_landmarks[idx]
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
                landmarks = [coord - base_coord for lm in hand_landmarks.landmark for coord, base_coord in zip([lm.x, lm.y, lm.z], [base_x, base_y, base_z])]

                hand_type = hand_info.classification[0].label
                if hand_type == 'Left': left_hand_landmarks = np.array(landmarks)
                elif hand_type == 'Right': right_hand_landmarks = np.array(landmarks)

            combined_landmarks = np.concatenate([left_hand_landmarks, right_hand_landmarks]).reshape(1, -1)
            combined_landmarks = np.nan_to_num(combined_landmarks)

            try:
                prediction_proba = self.model.predict_proba(combined_landmarks)[0]
                confidence = np.max(prediction_proba)
                predicted_label = self.model.classes_[np.argmax(prediction_proba)]

                if confidence >= self.CONFIDENCE_THRESHOLD:
                    self.prediction_buffer.append(predicted_label)
                    
                    if len(self.prediction_buffer) >= self.STABLE_THRESHOLD:
                        if all(p == predicted_label for p in list(self.prediction_buffer)[-self.STABLE_THRESHOLD:]):
                            stable_prediction = predicted_label
                    
                    display_prediction = predicted_label
                else:
                    self.prediction_buffer.clear()
            except Exception:
                self.prediction_buffer.clear()

        cv2.putText(frame, display_prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        bar_height = 50
        cv2.rectangle(frame, (0, frame.shape[0] - bar_height), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        
        return frame, stable_prediction

    def stop(self):
        self.running = False
        self.wait()

# --- 1.5. Sound Player Thread Class ---
class SoundPlayer(QThread):
    finished = Signal()

    def __init__(self, audio_file):
        super().__init__()
        self.audio_file = audio_file

    def run(self):
        """Loads and plays the audio file in a separate thread."""
        try:
            if not os.path.exists(self.audio_file):
                print("SoundPlayer: Audio file not found.")
                return

            pygame.mixer.init()
            pygame.mixer.music.load(self.audio_file)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            pygame.mixer.music.stop()
            pygame.mixer.music.unload() 
            
            if os.path.exists(self.audio_file):
                os.remove(self.audio_file)

        except Exception as e:
            print(f"SoundPlayer Error in thread: {e}")
        finally:
            self.finished.emit()
            pygame.mixer.quit() 


# --- 2. Main GUI Class (DYNAMIC WIDTH APPLIED) ---
class SignDetectorApp(QMainWindow):
    LANGUAGE_MAP = {
        "English": "en",
        "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Marathi": "mr", 
        "Bengali": "bn", "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml",
        "Odia": "or", "Punjabi": "pa", "Urdu": "ur", "Assamese": "as",
        "Sanskrit": "sa",
        "French": "fr", "Spanish": "es", "German": "de", "Japanese": "ja", 
        "Chinese (Simplified)": "zh-cn", "Russian": "ru", "Portuguese": "pt",
        "Arabic": "ar", "Korean": "ko", "Italian": "it", "Vietnamese": "vi",
        "Turkish": "tr", "Dutch": "nl", "Swedish": "sv", "Danish": "da",
        "Finnish": "fi", "Greek": "el", "Thai": "th",
    }

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.setWindowTitle("Real-Time Sequential Sign Recognition - RUNNING")
        
        self.setStyleSheet(f"background-color: {BG_COLOR_PRIMARY}; color: #FFFFFF;")
        self.setFont(QFont(FONT_NAME)) 
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_v_layout = QVBoxLayout(self.central_widget)
        self.main_v_layout.setContentsMargins(30, 30, 30, 30)
        self.main_v_layout.setSpacing(30) 

        self.worker = None
        self.current_word_text = ""
        self.sentence_display = deque()
        self.is_translated = False
        self.original_sentence = "" 
        self.current_display_text = "" 
        self.current_lang_code = "en" 
        self.sound_player_thread = None

        if TRANSLATOR_AVAILABLE: self.translator = Translator()
            
        # --- TOP ROW: Camera and Controls (DYNAMIC WIDTH) ---
        top_h_layout = QHBoxLayout()
        top_h_layout.setSpacing(30)
        
        # 1. Camera View Widget (DYNAMIC WIDTH)
        self.video_label = QLabel("Camera Display Here", alignment=Qt.AlignCenter)
        self.video_label.setMinimumSize(CAMERA_MIN_WIDTH, CAMERA_FIXED_HEIGHT) 
        self.video_label.setFixedHeight(CAMERA_FIXED_HEIGHT) 
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed) # Dynamic Width
        
        self.video_label.setStyleSheet(f"""
            background-color: {BG_COLOR_SECONDARY}; 
            border-radius: {BOX_CORNER_RADIUS_LARGE}; 
            color: #FFFFFF; 
            font-size: 20pt; 
            padding: 10px;
        """)
        
        # 2. Language Control Box (DYNAMIC WIDTH)
        self.language_control_box = QWidget()
        self.language_control_box.setMinimumWidth(LANGUAGE_BOX_MIN_WIDTH) 
        self.language_control_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum) # Dynamic Width
        
        self.language_control_box.setStyleSheet(f"""
            background-color: {BG_COLOR_SECONDARY};
            border-radius: {BOX_CORNER_RADIUS_SMALL};
            padding: 15px;
        """)
        
        controls_v_layout = QVBoxLayout(self.language_control_box)
        controls_v_layout.setSpacing(10)
        
        language_label = QLabel("Language:")
        language_label.setFont(QFont(FONT_NAME, 14)) 
        language_label.setStyleSheet("color: #FFFFFF; margin-bottom: 5px;")
        
        self.language_combo = QComboBox()
        self.language_list = sorted(list(self.LANGUAGE_MAP.keys()))
        self.language_combo.addItem("Select Target Language...")
        self.language_combo.addItems(self.language_list)
        
        completer = QCompleter(self.language_list)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.language_combo.setCompleter(completer)
        self.language_combo.setEditable(True)
        self.language_combo.lineEdit().setReadOnly(False)
        self.language_combo.lineEdit().setAlignment(Qt.AlignCenter)
        self.language_combo.setCurrentText("Select Target Language...")
        self.language_combo.setFixedHeight(CONTROL_BUTTON_HEIGHT)
        
        self.translate_btn = QPushButton("Convert Language")
        self.translate_btn.clicked.connect(self.translate_sentence)
        self.translate_btn.setEnabled(False)
        self.translate_btn.setFixedHeight(CONTROL_BUTTON_HEIGHT)
        
        self.tts_btn = QPushButton("Text to Voice 🔊")
        self.tts_btn.clicked.connect(self.text_to_voice)
        self.tts_btn.setEnabled(False)
        self.tts_btn.setFixedHeight(CONTROL_BUTTON_HEIGHT)
        
        controls_v_layout.addWidget(language_label)
        controls_v_layout.addWidget(self.language_combo)
        controls_v_layout.addWidget(self.translate_btn)
        controls_v_layout.addWidget(self.tts_btn)
        controls_v_layout.addStretch(1) 

        # DYNAMIC WIDTH ASSIGNMENT (3:1 Ratio for Camera:Language Box)
        top_h_layout.addWidget(self.video_label, 3)          
        top_h_layout.addWidget(self.language_control_box, 1) 
        self.main_v_layout.addLayout(top_h_layout)
        
        
        # --- MIDDLE ROW: Sentence Output Box (DYNAMIC WIDTH) ---
        
        self.sentence_output = QTextEdit()
        self.sentence_output.setReadOnly(True)
        self.sentence_output.setText("Recognized Sentence: ")
        self.sentence_output.setFixedHeight(SENTENCE_BOX_HEIGHT) 
        self.sentence_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        self.sentence_output.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_COLOR_SECONDARY};
                color: #FFFFFF;
                font-family: {FONT_NAME};
                font-size: {FONT_SIZE_BODY};
                border-radius: {BOX_CORNER_RADIUS_LARGE};
                padding: 15px; 
            }}
        """)
        
        self.main_v_layout.addWidget(self.sentence_output)

        
        # --- BOTTOM ROW: Buttons (DYNAMIC WIDTH) ---
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(30)
        
        self.start_btn = QPushButton("START CAMERA")
        self.clear_btn = QPushButton("CLEAR SENTENCE")
        self.quit_btn = QPushButton("QUIT")
        
        # Dynamic Width: Buttons share space equally (no fixed width calls)
        self.clear_btn.setEnabled(False)
        self.quit_btn.setEnabled(True)

        self.start_btn.clicked.connect(self.start_worker)
        self.clear_btn.clicked.connect(self.clear_text)
        self.quit_btn.clicked.connect(self.close)
        self.language_combo.currentTextChanged.connect(self.reset_translation_on_lang_change) 

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.quit_btn)
        self.main_v_layout.addLayout(button_layout)
        
        self.apply_button_styles()


    def apply_button_styles(self):
        # Global style for START/CLEAR buttons
        button_style = f"""
            QPushButton {{
                background-color: {BG_COLOR_BUTTON};
                color: #FFFFFF;
                font-family: {FONT_NAME};
                font-size: {FONT_SIZE_BODY};
                height: {BUTTON_HEIGHT}px;
                border-radius: {BOX_CORNER_RADIUS_LARGE};
                border: none;
            }}
            QPushButton:hover {{
                background-color: #7E7676; 
            }}
            QPushButton:pressed {{
                background-color: #4D4747; 
            }}
        """
        
        self.start_btn.setStyleSheet(button_style)
        self.clear_btn.setStyleSheet(button_style)
        
        # QUIT button specific style
        quit_style = f"""
            QPushButton {{
                background-color: {BG_COLOR_ACCENT_RED};
                color: #FFFFFF;
                font-family: {FONT_NAME};
                font-size: {FONT_SIZE_BODY};
                height: {BUTTON_HEIGHT}px;
                border-radius: {BOX_CORNER_RADIUS_LARGE};
                border: none;
            }}
            QPushButton:hover {{
                background-color: #902C2C; 
            }}
            QPushButton:pressed {{
                background-color: #550909; 
            }}
        """
        self.quit_btn.setStyleSheet(quit_style)

        # Convert Language Button
        convert_style = f"""
            QPushButton {{
                background-color: {BG_COLOR_BUTTON};
                color: #FFFFFF;
                font-family: {FONT_NAME};
                font-size: {FONT_SIZE_SMALL};
                height: {CONTROL_BUTTON_HEIGHT}px;
                border-radius: {BOX_CORNER_RADIUS_MEDIUM};
                border: none;
            }}
        """
        self.translate_btn.setStyleSheet(convert_style)
        
        # Text to Voice Button
        tts_style = f"""
            QPushButton {{
                background-color: {BG_COLOR_ACCENT_GREEN};
                color: #FFFFFF;
                font-family: {FONT_NAME};
                font-size: {FONT_SIZE_SMALL};
                height: {CONTROL_BUTTON_HEIGHT}px;
                border-radius: {BOX_CORNER_RADIUS_SMALL};
                border: none;
            }}
        """
        self.tts_btn.setStyleSheet(tts_style)
        
        # Language Select ComboBox style
        combo_style = f"""
            QComboBox {{
                background-color: {BG_COLOR_BUTTON};
                color: #FFFFFF;
                font-family: {FONT_NAME};
                font-size: {FONT_SIZE_SMALL};
                height: {CONTROL_BUTTON_HEIGHT}px;
                border-radius: {BOX_CORNER_RADIUS_SMALL};
                padding: 5px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {BG_COLOR_BUTTON};
                color: #FFFFFF;
                selection-background-color: {BG_COLOR_SECONDARY};
            }}
            QLineEdit {{
                background-color: {BG_COLOR_BUTTON}; 
                border: none; 
                color: #FFFFFF;
            }}
        """
        self.language_combo.setStyleSheet(combo_style)
        self.language_combo.lineEdit().setStyleSheet(f"""
            QLineEdit {{
                background-color: {BG_COLOR_BUTTON};
                color: #FFFFFF;
                border: none;
                font-size: {FONT_SIZE_SMALL};
            }}
        """)


    # --- Functionality Methods (Unchanged) ---
    def translate_sentence(self):
        if not TRANSLATOR_AVAILABLE:
            self.sentence_output.setText("Error: Translation library not installed.")
            return

        target_lang_name = self.language_combo.currentText()
        if target_lang_name not in self.LANGUAGE_MAP:
            self.sentence_output.setText("Error: Please select a valid language from the list.")
            return

        target_lang_code = self.LANGUAGE_MAP.get(target_lang_name, 'en') 

        try:
            text_to_translate = self.original_sentence
            
            if not text_to_translate:
                return 

            if target_lang_name == "English":
                translated_text = self.original_sentence
                display_name = "English"
                self.is_translated = False 
            else:
                translation = self.translator.translate(text_to_translate, dest=target_lang_code)
                translated_text = translation.text
                display_name = target_lang_name
                self.is_translated = True
            
            self.current_display_text = translated_text
            self.current_lang_code = target_lang_code 
            
            self.sentence_output.setText(f"Recognized Sentence ({display_name}): {self.current_display_text}")
            
        except Exception as e:
            self.sentence_output.setText(f"Translation Error: {e}")

    def text_to_voice(self):
        if not TTS_AVAILABLE:
            self.sentence_output.setText("Error: Text-to-Voice libraries not installed.")
            return
            
        if not self.current_display_text:
            return

        try:
            if TTS_AVAILABLE and pygame.mixer.get_init():
                 pygame.mixer.music.stop()
                 pygame.mixer.music.unload()

            self.remove_temp_file(TTS_TEMP_FILE)
            
            tts = gTTS(text=self.current_display_text, lang=self.current_lang_code) 
            tts.save(TTS_TEMP_FILE)
            
        except Exception as e:
            self.sentence_output.setText(f"TTS File Save Error (Check permissions/file lock): {e}")
            return

        if hasattr(self, 'sound_player_thread') and self.sound_player_thread and self.sound_player_thread.isRunning():
            self.sound_player_thread.terminate()
            self.sound_player_thread.wait()

        self.sound_player_thread = SoundPlayer(TTS_TEMP_FILE)
        self.sound_player_thread.start()


    def remove_temp_file(self, filename):
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except PermissionError as e:
            print(f"Cleanup Warning: Could not remove temporary file {filename}. Error: {e}")


    def reset_translation_on_lang_change(self):
        if not self.is_translated or not self.original_sentence:
            return
        
        self.current_display_text = self.original_sentence
        self.current_lang_code = "en"
        self.is_translated = False
        self.sentence_output.setText(f"Recognized Sentence: {self.current_display_text}")

    @Slot(QImage)
    def update_frame(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    @Slot(str)
    def append_word(self, word):
        if self.sentence_display and self.sentence_display[-1] == word:
             return 
            
        self.is_translated = False 
        
        self.sentence_display.append(word)
        self.original_sentence = ' '.join(self.sentence_display)
        self.current_display_text = self.original_sentence
        self.current_lang_code = "en" 
        
        self.sentence_output.setText(f"Recognized Sentence: {self.current_display_text}")
        self.sentence_output.verticalScrollBar().setValue(self.sentence_output.verticalScrollBar().maximum())
        
        self.translate_btn.setEnabled(True)
        self.tts_btn.setEnabled(True)

    def start_worker(self):
        if self.worker is None or not self.worker.isRunning():
            self.worker = VideoWorker(self.model)
            self.worker.frame_ready.connect(self.update_frame)
            self.worker.word_ready.connect(self.append_word)
            self.worker.start()
            
            self.start_btn.setEnabled(False)
            self.clear_btn.setEnabled(True)
            self.setWindowTitle("Real-Time Sequential Sign Recognition - RUNNING")


    def clear_text(self):
        self.sentence_display.clear()
        self.is_translated = False
        self.original_sentence = ""
        self.current_display_text = ""
        self.current_lang_code = "en"
        
        self.sentence_output.setText("Recognized Sentence: ")
        
        self.translate_btn.setEnabled(False)
        self.tts_btn.setEnabled(False)
        
        if self.worker:
            self.worker.prediction_buffer.clear()
            self.worker.last_stable_prediction = None
            
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        
        if hasattr(self, 'sound_player_thread') and self.sound_player_thread and self.sound_player_thread.isRunning():
            self.sound_player_thread.terminate()

        if TTS_AVAILABLE and pygame.mixer.get_init():
            pygame.mixer.quit()
        
        self.remove_temp_file(TTS_TEMP_FILE) 
        
        event.accept()

# --- 3. Main Execution ---
if __name__ == "__main__":
    
    if not os.path.exists(MODEL_FILE):
        print("FATAL ERROR: Model not found. Cannot start application.")
        sys.exit(1)
        
    with open(MODEL_FILE, 'rb') as f:
        global_model = pickle.load(f)

    app = QApplication(sys.argv)
    app.setFont(QFont("Roboto"))
    
    window = SignDetectorApp(global_model)
    window.setMinimumSize(1000, 750) 
    window.show()
    sys.exit(app.exec())