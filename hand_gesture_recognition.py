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

# Initialize MediaPipe Hands for hand landmark detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)  # Track up to 2 hands with 70% confidence
mp_draw = mp.solutions.drawing_utils  # Utility for drawing landmarks on video frames

# Define gesture labels and file paths
GESTURES = ['open_hand', 'closed_fist', 'peace_sign', 'thumbs_up', 'pointing']  # List of gestures to recognize
DATA_DIR = 'gesture_data'  # Directory to store gesture data
MODEL_PATH = 'gesture_model.pkl'  # Path to save trained model
SELECTOR_PATH = 'selector.pkl'  # Path to save feature selector
SEQUENCE_PATTERNS = [['open_hand', 'closed_fist'], ['peace_sign', 'thumbs_up']]  # Gesture sequences to detect

# Create directories for storing gesture data
def create_data_dirs():
    """Create directories for each gesture if they don't exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)  # Create root data directory
    for gesture in GESTURES:
        gesture_path = os.path.join(DATA_DIR, gesture)
        if not os.path.exists(gesture_path):
            os.makedirs(gesture_path)  # Create subdirectory for each gesture

# Normalize hand landmarks relative to the wrist
def normalize_landmarks(landmarks):
    """Normalize landmarks by subtracting wrist coordinates to make data position-invariant."""
    wrist = landmarks[0:3]  # Wrist is the first landmark (x, y, z)
    normalized = []
    for i in range(0, len(landmarks), 3):
        x, y, z = landmarks[i:i+3]  # Extract x, y, z coordinates
        normalized.extend([x - wrist[0], y - wrist[1], z - wrist[2]])  # Subtract wrist coordinates
    return normalized

# Augment landmarks with random rotation
def augment_landmarks(landmarks):
    """Apply random 2D rotation to landmarks to simulate variations and improve model robustness."""
    angle = np.random.uniform(-15, 15)  # Random rotation angle between -15 and 15 degrees
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))  # Compute cosine and sine
    augmented = []
    for i in range(0, len(landmarks), 3):
        x, y, z = landmarks[i:i+3]
        x_rot = x * cos_a - y * sin_a  # Rotate x coordinate
        y_rot = x * sin_a + y * cos_a  # Rotate y coordinate
        augmented.extend([x_rot, y_rot, z])  # Keep z unchanged
    return augmented

# Collect gesture data from webcam
def collect_data(gesture, status_label, sample_label, video_label, progress_bar, max_samples=200):
    """Collect hand landmark data for a specific gesture and save it to files."""
    cap = cv2.VideoCapture(0)  # Open default webcam
    sample_count = 0  # Track number of samples collected
    
    # Update GUI with initial status
    status_label.setText(f"Collecting: {gesture}. Press 'Start' to capture.")
    progress_bar.setMaximum(max_samples)  # Set progress bar limit
    
    def capture():
        """Inner function to handle data collection in a separate thread."""
        nonlocal sample_count
        while sample_count < max_samples and not stop_collection[0]:
            ret, frame = cap.read()  # Read frame from webcam
            if not ret:
                continue  # Skip if frame capture fails
                
            frame = cv2.flip(frame, 1)  # Flip frame horizontally for mirror effect
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for MediaPipe
            result = hands.process(frame_rgb)  # Process frame to detect hands
            
            # Process detected hand landmarks
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Draw landmarks
                    
                    # Extract and normalize landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])  # Collect x, y, z coordinates
                    landmarks = normalize_landmarks(landmarks)  # Normalize relative to wrist
                    
                    # Save landmarks to a unique file
                    np.save(os.path.join(DATA_DIR, gesture, f'{uuid.uuid4()}.npy'), landmarks)
                    sample_count += 1
                    
                    # Update GUI
                    sample_label.setText(f"Samples: {sample_count}/{max_samples}")
                    progress_bar.setValue(sample_count)
                    status_label.setText(f"Capturing: {gesture}")
                    break  # Process only one hand per frame
            
            # Display video feed in GUI
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
            qimg = QImage(img.data, w, h, w * c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(480, 360, Qt.KeepAspectRatio)
            video_label.setPixmap(pixmap)
            
        # Cleanup after collection
        cap.release()
        status_label.setText("Collection stopped.")
        sample_label.setText("")
        progress_bar.setValue(0)
    
    def start_capture():
        """Start data collection in a separate thread."""
        stop_collection[0] = False
        threading.Thread(target=capture, daemon=True).start()
    
    def stop_capture():
        """Stop data collection."""
        stop_collection[0] = True
    
    # Initialize stop flag and connect buttons
    stop_collection = [False]  # Mutable list to allow modification in inner functions
    start_button.clicked.connect(start_capture)
    stop_button.clicked.connect(stop_capture)

# Train the gesture recognition model
def train_model(status_label):
    """Train a Random Forest classifier on collected gesture data."""
    status_label.setText("Training model...")
    X, y = [], []  # Lists for features and labels
    
    # Load data for each gesture
    for gesture_idx, gesture in enumerate(GESTURES):
        gesture_path = os.path.join(DATA_DIR, gesture)
        for file_name in os.listdir(gesture_path):
            landmarks = np.load(os.path.join(gesture_path, file_name))  # Load landmark data
            X.append(landmarks)  # Add original data
            y.append(gesture_idx)  # Add corresponding label
            for _ in range(2):  # Augment data twice
                X.append(augment_landmarks(landmarks))
                y.append(gesture_idx)
    
    # Check if data exists
    if len(X) == 0:
        status_label.setText("No data found! Collect data first.")
        return
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Select top 30 features using ANOVA F-value
    selector = SelectKBest(f_classif, k=30)
    X = selector.fit_transform(X, y)
    
    # Train Random Forest classifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Save model and feature selector
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SELECTOR_PATH, 'wb') as f:
        pickle.dump(selector, f)
    
    status_label.setText("Model trained and saved!")

# Test the model in real-time
def test_model(status_label, confidence_label, video_label, fps_label):
    """Perform real-time gesture recognition using the trained model."""
    cap = cv2.VideoCapture(0)  # Open webcam
    
    # Check if model and selector exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SELECTOR_PATH):
        status_label.setText("Model or selector not found! Train first.")
        return
    
    # Load model and selector
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SELECTOR_PATH, 'rb') as f:
        selector = pickle.load(f)
    
    gesture_buffer = []  # Buffer to store recent gestures for sequence detection
    start_time = time.time()  # For FPS calculation
    frame_count = 0  # Count frames for FPS
    
    def predict():
        """Inner function to handle real-time prediction in a separate thread."""
        nonlocal frame_count, start_time
        while not stop_testing[0]:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)  # Flip frame for mirror effect
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            result = hands.process(frame_rgb)  # Detect hands
            
            gesture = "No gesture"  # Default if no hand detected
            confidence = 0.0
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Draw landmarks
                    
                    # Extract and normalize landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    landmarks = normalize_landmarks(landmarks)
                    
                    # Predict gesture
                    landmarks = selector.transform([landmarks])[0]  # Apply feature selection
                    probs = model.predict_proba([landmarks])[0]  # Get prediction probabilities
                    prediction = np.argmax(probs)  # Get most likely class
                    confidence = probs[prediction] * 100  # Convert to percentage
                    gesture = GESTURES[prediction]  # Map to gesture name
                    break  # Process only one hand
            
            # Update gesture buffer for sequence detection
            gesture_buffer.append(gesture)
            if len(gesture_buffer) > 5:
                gesture_buffer.pop(0)  # Keep only last 5 gestures
            for pattern in SEQUENCE_PATTERNS:
                if gesture_buffer[-len(pattern):] == pattern:
                    status_label.setText(f"Sequence: {' -> '.join(pattern)}")  # Display detected sequence
                    break
            else:
                status_label.setText(f"Detected: {gesture}")  # Display single gesture
            
            # Update video feed
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
            qimg = QImage(img.data, w, h, w * c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(480, 360, Qt.KeepAspectRatio)
            video_label.setPixmap(pixmap)
            
            # Update confidence and FPS
            confidence_label.setText(f"Confidence: {confidence:.2f}%")
            frame_count += 1
            if time.time() - start_time >= 1:
                fps = frame_count / (time.time() - start_time)
                fps_label.setText(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()
        
        # Cleanup after testing
        cap.release()
        status_label.setText("Testing stopped.")
        confidence_label.setText("")
        fps_label.setText("")
    
    def stop_test():
        """Stop real-time testing."""
        stop_testing[0] = True
    
    # Initialize stop flag and connect stop button
    stop_testing = [False]
    stop_button.clicked.connect(stop_test)
    threading.Thread(target=predict, daemon=True).start()

# PyQt5 GUI
class MainWindow(QMainWindow):
    def __init__(self):
        """Initialize the main window and set up the GUI."""
        super().__init__()
        self.setWindowTitle("Hand Gesture Recognition")  # Window title
        self.setGeometry(100, 100, 900, 600)  # Window size and position
        self.is_dark = False  # Track theme (light/dark)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Top layout for live view and prediction panels
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)
        
        # Live view widget
        self.live_widget = QWidget()
        live_layout = QVBoxLayout()
        self.live_widget.setLayout(live_layout)
        self.live_widget.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; color: black;")
        
        live_label = QLabel("Live View")  # Label for video feed
        live_label.setFont(QFont("Arial", 12, QFont.Bold))
        live_label.setAlignment(Qt.AlignCenter)
        live_layout.addWidget(live_label)
        
        self.video_label = QLabel()  # Display webcam feed
        self.video_label.setFixedSize(480, 360)
        self.video_label.setAlignment(Qt.AlignCenter)
        live_layout.addWidget(self.video_label)
        
        top_layout.addWidget(self.live_widget)
        
        # Prediction widget
        self.pred_widget = QWidget()
        pred_layout = QVBoxLayout()
        self.pred_widget.setLayout(pred_layout)
        self.pred_widget.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; color: black;")
        
        pred_label = QLabel("Prediction")  # Label for prediction info
        pred_label.setFont(QFont("Arial", 12, QFont.Bold))
        pred_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(pred_label)
        
        self.status_label = QLabel("Welcome! Select an action.")  # Display status messages
        self.status_label.setFont(QFont("Arial", 11))
        self.status_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self.status_label)
        
        self.confidence_label = QLabel("")  # Display prediction confidence
        self.confidence_label.setFont(QFont("Arial", 11))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self.confidence_label)
        
        self.sample_label = QLabel("")  # Display sample count during collection
        self.sample_label.setFont(QFont("Arial", 11))
        self.sample_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self.sample_label)
        
        self.progress_bar = QProgressBar()  # Progress bar for data collection
        pred_layout.addWidget(self.progress_bar)
        
        self.fps_label = QLabel("")  # Display FPS during testing
        self.fps_label.setFont(QFont("Arial", 11))
        self.fps_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self.fps_label)
        
        top_layout.addWidget(self.pred_widget)
        
        # Button panel
        self.button_widget = QWidget()
        button_layout = QHBoxLayout()
        self.button_widget.setLayout(button_layout)
        self.button_widget.setStyleSheet("background-color: #e0e0e0; padding: 10px;")
        main_layout.addWidget(self.button_widget)
        
        # Gesture selection dropdown
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
        
        # Global buttons for starting/stopping actions
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
        
        # Train model button
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
        
        # Test model button
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
        
        # Input field for new gestures
        self.gesture_input = QLineEdit()
        self.gesture_input.setPlaceholderText("Enter new gesture")
        self.gesture_input.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 4px;")
        button_layout.addWidget(self.gesture_input)
        
        # Add gesture button
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
        
        # Theme toggle button
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
        
        # Quit button
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
        """Start data collection for the selected gesture."""
        gesture = self.gesture_combo.currentText()
        collect_data(gesture, self.status_label, self.sample_label, self.video_label, self.progress_bar)
    
    def add_gesture(self):
        """Add a new gesture to the list and create its directory."""
        new_gesture = self.gesture_input.text().strip()
        if new_gesture and new_gesture not in GESTURES:
            GESTURES.append(new_gesture)  # Add to global gesture list
            self.gesture_combo.addItem(new_gesture)  # Update dropdown
            os.makedirs(os.path.join(DATA_DIR, new_gesture), exist_ok=True)  # Create directory
            self.status_label.setText(f"Added gesture: {new_gesture}")
            self.gesture_input.clear()
        else:
            self.status_label.setText("Invalid or duplicate gesture name!")
    
    def toggle_theme(self):
        """Toggle between light and dark themes."""
        self.is_dark = not self.is_dark
        if self.is_dark:
            # Apply dark theme
            self.live_widget.setStyleSheet("background-color: #424242; border: 1px solid #616161; color: white;")
            self.pred_widget.setStyleSheet("background-color: #424242; border: 1px solid #616161; color: white;")
            self.button_widget.setStyleSheet("background-color: #616161; padding: 10px;")
        else:
            # Apply light theme
            self.live_widget.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; color: black;")
            self.pred_widget.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; color: black;")
            self.button_widget.setStyleSheet("background-color: #e0e0e0; padding: 10px;")

# Run the application
if __name__ == '__main__':
    create_data_dirs()  # Create directories for gesture data
    app = QApplication(sys.argv)  # Initialize PyQt5 application
    window = MainWindow()  # Create main window
    window.show()  # Show window
    sys.exit(app.exec_())  # Start event loop