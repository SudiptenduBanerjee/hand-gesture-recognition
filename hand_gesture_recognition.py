import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import tkinter as tk
from PIL import Image, ImageTk
import threading
import uuid

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Gesture labels
GESTURES = ['open_hand', 'closed_fist', 'peace_sign', 'thumbs_up', 'pointing']
DATA_DIR = 'gesture_data'
MODEL_PATH = 'gesture_model.pkl'

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
    wrist = landmarks[0:3]  # Wrist landmark (x, y, z)
    normalized = []
    for i in range(0, len(landmarks), 3):
        x, y, z = landmarks[i:i+3]
        normalized.extend([x - wrist[0], y - wrist[1], z - wrist[2]])
    return normalized

# Collect gesture data
def collect_data(gesture, status_label, sample_label, max_samples=200):
    cap = cv2.VideoCapture(0)
    sample_count = 0
    
    status_label.config(text=f"Collecting: {gesture}. Press 'Start' to capture.")
    
    def capture():
        nonlocal sample_count
        while sample_count < max_samples:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Extract and normalize landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    landmarks = normalize_landmarks(landmarks)
                    
                    # Save landmarks
                    np.save(os.path.join(DATA_DIR, gesture, f'{uuid.uuid4()}.npy'), landmarks)
                    sample_count += 1
                    
                    sample_label.config(text=f"Samples: {sample_count}/{max_samples}")
                    status_label.config(text=f"Capturing: {gesture}")
            
            # Update video feed
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            
            if stop_collection:
                break
        
        cap.release()
        status_label.config(text="Collection stopped.")
        sample_label.config(text="")
    
    def start_capture():
        nonlocal stop_collection
        stop_collection = False
        threading.Thread(target=capture, daemon=True).start()
    
    def stop_capture():
        nonlocal stop_collection
        stop_collection = True
    
    stop_collection = False
    start_button.config(command=start_capture)
    stop_button.config(command=stop_capture)

# Train the model
def train_model(status_label):
    status_label.config(text="Training model...")
    X, y = [], []
    
    for gesture_idx, gesture in enumerate(GESTURES):
        gesture_path = os.path.join(DATA_DIR, gesture)
        for file_name in os.listdir(gesture_path):
            landmarks = np.load(os.path.join(gesture_path, file_name))
            X.append(landmarks)
            y.append(gesture_idx)
    
    if len(X) == 0:
        status_label.config(text="No data found! Collect data first.")
        return
    
    X = np.array(X)
    y = np.array(y)
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    status_label.config(text="Model trained and saved!")

# Test the model in real-time
def test_model(status_label):
    cap = cv2.VideoCapture(0)
    
    # Load the model
    if not os.path.exists(MODEL_PATH):
        status_label.config(text="Model not found! Train first.")
        return
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    def predict():
        while not stop_testing:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
            
            gesture = "No gesture"
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Extract and normalize landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    landmarks = normalize_landmarks(landmarks)
                    
                    # Predict gesture
                    prediction = model.predict([landmarks])[0]
                    gesture = GESTURES[prediction]
            
            # Update video feed
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            
            status_label.config(text=f"Detected: {gesture}")
        
        cap.release()
        status_label.config(text="Testing stopped.")
    
    def stop_test():
        nonlocal stop_testing
        stop_testing = True
    
    stop_testing = False
    stop_button.config(command=stop_test)
    threading.Thread(target=predict, daemon=True).start()

# Tkinter UI
root = tk.Tk()
root.title("Hand Gesture Recognition")
root.geometry("800x600")

# Video feed
video_label = tk.Label(root)
video_label.pack(pady=10)

# Status and sample count
status_label = tk.Label(root, text="Welcome! Select an action.", font=("Arial", 12))
status_label.pack(pady=5)
sample_label = tk.Label(root, text="", font=("Arial", 10))
sample_label.pack(pady=5)

# Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Gesture selection for data collection
gesture_var = tk.StringVar(value=GESTURES[0])
gesture_menu = tk.OptionMenu(button_frame, gesture_var, *GESTURES)
gesture_menu.config(width=15)
gesture_menu.grid(row=0, column=0, padx=5)

start_button = tk.Button(button_frame, text="Start Collection", width=15)
start_button.grid(row=0, column=1, padx=5)

stop_button = tk.Button(button_frame, text="Stop", width=15)
stop_button.grid(row=0, column=2, padx=5)

train_button = tk.Button(button_frame, text="Train Model", width=15, 
                        command=lambda: train_model(status_label))
train_button.grid(row=1, column=0, padx=5, pady=5)

test_button = tk.Button(button_frame, text="Test Model", width=15, 
                       command=lambda: test_model(status_label))
test_button.grid(row=1, column=1, padx=5, pady=5)

quit_button = tk.Button(button_frame, text="Quit", width=15, command=root.quit)
quit_button.grid(row=1, column=2, padx=5, pady=5)

# Create data directories
create_data_dirs()

# Start collection when gesture is selected
def start_collection():
    collect_data(gesture_var.get(), status_label, sample_label)

start_button.config(command=start_collection)

# Run the application
root.mainloop()