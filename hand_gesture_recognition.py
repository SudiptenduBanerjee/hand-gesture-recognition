import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import os
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QProgressBar, QLineEdit)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt
import uuid
import threading
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Gesture labels
GESTURES = ['open_hand', 'closed_fist', 'peace_sign', 'thumbs_up', 'pointing']
DATA_DIR = 'gesture_data'
MODEL_PATH = 'gesture_model.pkl'
SELECTOR_PATH = 'selector.pkl'
SEQUENCE_PATTERNS = [['open_hand', 'closed_fist'], ['peace_sign', 'thumbs_up']]

# Create directories for data collection
def create_data_dirs():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    for gesture in GESTURES:
        gesture_path = os.path.join(DATA_DIR, gesture)
        if not os.path.exists(gesture_path):
            os.makedirs(gesture_path)

# Normalize landmarks relative to wrist
def normalize_landmarks(landmarks):
    wrist = landmarks[0:3]
    normalized = []
    for i in range(0, len(landmarks), 3):
        x, y, z = landmarks[i:i+3]
        normalized.extend([x - wrist[0], y - wrist[1], z - wrist[2]])
    return normalized

# Data augmentation
def augment_landmarks(landmarks):
    angle = np.random.uniform(-15, 15)
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    augmented = []
    for i in range(0, len(landmarks), 3):
        x, y, z = landmarks[i:i+3]
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        augmented.extend([x_rot, y_rot, z])
    return augmented

# Collect gesture data
def collect_data(gesture, status_label, sample_label, video_label, progress_bar, max_samples=200):
    cap = cv2.VideoCapture(0)
    sample_count = 0
    
    status_label.setText(f"Collecting: {gesture}. Press 'Start' to capture.")
    progress_bar.setMaximum(max_samples)
    
    def capture():
        nonlocal sample_count
        while sample_count < max_samples and not stop_collection[0]:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    landmarks = normalize_landmarks(landmarks)
                    
                    np.save(os.path.join(DATA_DIR, gesture, f'{uuid.uuid4()}.npy'), landmarks)
                    sample_count += 1
                    
                    sample_label.setText(f"Samples: {sample_count}/{max_samples}")
                    progress_bar.setValue(sample_count)
                    status_label.setText(f"Capturing: {gesture}")
                    break
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
            qimg = QImage(img.data, w, h, w * c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(480, 360, Qt.KeepAspectRatio)
            video_label.setPixmap(pixmap)
            
        cap.release()
        status_label.setText("Collection stopped.")
        sample_label.setText("")
        progress_bar.setValue(0)
    
    def start_capture():
        stop_collection[0] = False
        threading.Thread(target=capture, daemon=True).start()
    
    def stop_capture():
        stop_collection[0] = True
    
    stop_collection = [False]
    start_button.clicked.connect(start_capture)
    stop_button.clicked.connect(stop_capture)

# Train the model
def train_model(status_label):
    status_label.setText("Training model...")
    X, y = [], []
    
    for gesture_idx, gesture in enumerate(GESTURES):
        gesture_path = os.path.join(DATA_DIR, gesture)
        for file_name in os.listdir(gesture_path):
            landmarks = np.load(os.path.join(gesture_path, file_name))
            X.append(landmarks)
            y.append(gesture_idx)  # Fixed: Changed 'label' to 'gesture_idx'
            for _ in range(2):
                X.append(augment_landmarks(landmarks))
                y.append(gesture_idx)
    
    if len(X) == 0:
        status_label.setText("No data found! Collect data first.")
        return
    
    X = np.array(X)
    y = np.array(y)
    
    selector = SelectKBest(f_classif, k=30)
    X = selector.fit_transform(X, y)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SELECTOR_PATH, 'wb') as f:
        pickle.dump(selector, f)
    
    status_label.setText("Model trained and saved!")

# Test the model in real-time
def test_model(status_label, confidence_label, video_label, fps_label):
    cap = cv2.VideoCapture(0)
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SELECTOR_PATH):
        status_label.setText("Model or selector not found! Train first.")
        return
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SELECTOR_PATH, 'rb') as f:
        selector = pickle.load(f)
    
    gesture_buffer = []
    start_time = time.time()
    frame_count = 0
    
    def predict():
        nonlocal frame_count, start_time
        while not stop_testing[0]:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
            
            gesture = "No gesture"
            confidence = 0.0
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    landmarks = normalize_landmarks(landmarks)
                    
                    landmarks = selector.transform([landmarks])[0]
                    probs = model.predict_proba([landmarks])[0]
                    prediction = np.argmax(probs)
                    confidence = probs[prediction] * 100
                    gesture = GESTURES[prediction]
                    break
            
            gesture_buffer.append(gesture)
            if len(gesture_buffer) > 5:
                gesture_buffer.pop(0)
            for pattern in SEQUENCE_PATTERNS:
                if gesture_buffer[-len(pattern):] == pattern:
                    status_label.setText(f"Sequence: {' -> '.join(pattern)}")
                    break
            else:
                status_label.setText(f"Detected: {gesture}")
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
            qimg = QImage(img.data, w, h, w * c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(480, 360, Qt.KeepAspectRatio)
            video_label.setPixmap(pixmap)
            
            confidence_label.setText(f"Confidence: {confidence:.2f}%")
            
            frame_count += 1
            if time.time() - start_time >= 1:
                fps = frame_count / (time.time() - start_time)
                fps_label.setText(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()
        
        cap.release()
        status_label.setText("Testing stopped.")
        confidence_label.setText("")
        fps_label.setText("")
    
    def stop_test():
        stop_testing[0] = True
    
    stop_testing = [False]
    stop_button.clicked.connect(stop_test)
    threading.Thread(target=predict, daemon=True).start()

# PyQt5 UI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Gesture Recognition")
        self.setGeometry(100, 100, 900, 600)
        self.is_dark = False
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)
        
        self.live_widget = QWidget()
        live_layout = QVBoxLayout()
        self.live_widget.setLayout(live_layout)
        self.live_widget.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; color: black;")
        
        live_label = QLabel("Live View")
        live_label.setFont(QFont("Arial", 12, QFont.Bold))
        live_label.setAlignment(Qt.AlignCenter)
        live_layout.addWidget(live_label)
        
        self.video_label = QLabel()
        self.video_label.setFixedSize(480, 360)
        self.video_label.setAlignment(Qt.AlignCenter)
        live_layout.addWidget(self.video_label)
        
        top_layout.addWidget(self.live_widget)
        
        self.pred_widget = QWidget()
        pred_layout = QVBoxLayout()
        self.pred_widget.setLayout(pred_layout)
        self.pred_widget.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; color: black;")
        
        pred_label = QLabel("Prediction")
        pred_label.setFont(QFont("Arial", 12, QFont.Bold))
        pred_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(pred_label)
        
        self.status_label = QLabel("Welcome! Select an action.")
        self.status_label.setFont(QFont("Arial", 11))
        self.status_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self.status_label)
        
        self.confidence_label = QLabel("")
        self.confidence_label.setFont(QFont("Arial", 11))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self.confidence_label)
        
        self.sample_label = QLabel("")
        self.sample_label.setFont(QFont("Arial", 11))
        self.sample_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self.sample_label)
        
        self.progress_bar = QProgressBar()
        pred_layout.addWidget(self.progress_bar)
        
        self.fps_label = QLabel("")
        self.fps_label.setFont(QFont("Arial", 11))
        self.fps_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self.fps_label)
        
        top_layout.addWidget(self.pred_widget)
        
        self.button_widget = QWidget()
        button_layout = QHBoxLayout()
        self.button_widget.setLayout(button_layout)
        self.button_widget.setStyleSheet("background-color: #e0e0e0; padding: 10px;")
        main_layout.addWidget(self.button_widget)
        
        self.gesture_combo = QComboBox()
        self.gesture_combo.addItems(GESTURES)
        self.gesture_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
        """)
        button_layout.addWidget(self.gesture_combo)
        
        global start_button, stop_button
        start_button = QPushButton("Start Collection")
        start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        start_button.clicked.connect(self.start_collection)
        button_layout.addWidget(start_button)
        
        stop_button = QPushButton("Stop")
        stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        button_layout.addWidget(stop_button)
        
        train_button = QPushButton("Train Model")
        train_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        train_button.clicked.connect(lambda: train_model(self.status_label))
        button_layout.addWidget(train_button)
        
        test_button = QPushButton("Test Model")
        test_button.setStyleSheet("""
            QPushButton {
                background-color: #FFC107;
                color: black;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #FFB300;
            }
        """)
        test_button.clicked.connect(lambda: test_model(self.status_label, self.confidence_label, self.video_label, self.fps_label))
        button_layout.addWidget(test_button)
        
        self.gesture_input = QLineEdit()
        self.gesture_input.setPlaceholderText("Enter new gesture")
        self.gesture_input.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 4px;")
        button_layout.addWidget(self.gesture_input)
        
        add_gesture_button = QPushButton("Add Gesture")
        add_gesture_button.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        add_gesture_button.clicked.connect(self.add_gesture)
        button_layout.addWidget(add_gesture_button)
        
        theme_button = QPushButton("Toggle Theme")
        theme_button.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
        """)
        theme_button.clicked.connect(self.toggle_theme)
        button_layout.addWidget(theme_button)
        
        quit_button = QPushButton("Quit")
        quit_button.setStyleSheet("""
            QPushButton {
                background-color: #9E9E9E;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #757575;
            }
        """)
        quit_button.clicked.connect(self.close)
        button_layout.addWidget(quit_button)
    
    def start_collection(self):
        gesture = self.gesture_combo.currentText()
        collect_data(gesture, self.status_label, self.sample_label, self.video_label, self.progress_bar)
    
    def add_gesture(self):
        new_gesture = self.gesture_input.text().strip()
        if new_gesture and new_gesture not in GESTURES:
            GESTURES.append(new_gesture)
            self.gesture_combo.addItem(new_gesture)
            os.makedirs(os.path.join(DATA_DIR, new_gesture), exist_ok=True)
            self.status_label.setText(f"Added gesture: {new_gesture}")
            self.gesture_input.clear()
        else:
            self.status_label.setText("Invalid or duplicate gesture name!")
    
    def toggle_theme(self):
        self.is_dark = not self.is_dark
        if self.is_dark:
            self.live_widget.setStyleSheet("background-color: #424242; border: 1px solid #616161; color: white;")
            self.pred_widget.setStyleSheet("background-color: #424242; border: 1px solid #616161; color: white;")
            self.button_widget.setStyleSheet("background-color: #616161; padding: 10px;")
        else:
            self.live_widget.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; color: black;")
            self.pred_widget.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; color: black;")
            self.button_widget.setStyleSheet("background-color: #e0e0e0; padding: 10px;")

# Run the application
if __name__ == '__main__':
    create_data_dirs()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())