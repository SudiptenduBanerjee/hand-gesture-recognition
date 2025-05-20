Hand Gesture Recognition System
Welcome to the Hand Gesture Recognition System, an innovative project that brings intuitive hand gesture control to your fingertips! Powered by computer vision and machine learning, this application uses MediaPipe for real-time hand tracking, scikit-learn for gesture classification, and a sleek PyQt5-based GUI for an immersive user experience. Whether you're collecting gesture data, training a model, or testing gestures in real-time, the vibrant and customizable interface makes it seamless and engaging.
🚀 Features

Real-Time Gesture Recognition: Detect gestures like open hand, closed fist, peace sign, thumbs up, and pointing with high accuracy.
Sequence Detection: Recognize predefined gesture sequences (e.g., open hand → closed fist) for advanced interactions.
Data Collection: Easily collect hand landmark data using your webcam, with live video feedback and progress tracking.
Model Training: Train a Random Forest classifier with augmented data for robust gesture recognition.
Immersive UI: A modern PyQt5 interface with:
Live webcam feed with hand landmark visualization.
Real-time status updates, confidence scores, and FPS display.
Light and dark theme toggle for personalized aesthetics.
Intuitive controls for data collection, training, testing, and gesture management.


Custom Gestures: Add new gestures dynamically through the GUI.
Data Augmentation: Enhance model robustness with randomized rotations of hand landmarks.
Responsive Design: Threaded operations ensure the GUI remains smooth during intensive tasks.

📋 Prerequisites
To run this project, ensure you have the following installed:

Python 3.8+
Dependencies (install via pip):pip install opencv-python mediapipe numpy scikit-learn PyQt5


Hardware:
A webcam for capturing hand gestures.
Sufficient storage for gesture data (saved as .npy files).


Operating System: Tested on Windows; should work on macOS and Linux with minor adjustments.

🛠️ Installation

Clone the Repository:
git clone https://github.com/your-repo/hand-gesture-recognition.git
cd hand-gesture-recognition


Install Dependencies:
pip install -r requirements.txt

Create a requirements.txt with:
opencv-python
mediapipe
numpy
scikit-learn
PyQt5


Run the Application:
python gesture_recognition.py

Replace gesture_recognition.py with the name of your main script file.


🎮 Usage

Launch the Application:

Run the script to open the GUI.
The window displays a live video feed, prediction panel, and control buttons.


Collect Gesture Data:

Select a gesture from the dropdown (e.g., "open_hand").
Click Start Collection to begin capturing hand landmarks via webcam.
View the live feed with drawn landmarks and track progress via the progress bar.
Click Stop to end collection (default: 200 samples per gesture).


Train the Model:

Click Train Model to process collected data, augment it, and train a Random Forest classifier.
The model and feature selector are saved as gesture_model.pkl and selector.pkl.


Test in Real-Time:

Click Test Model to start real-time gesture recognition.
See detected gestures, confidence scores, and FPS in the GUI.
Gesture sequences (e.g., peace sign → thumbs up) are highlighted when detected.
Click Stop to end testing.


Add New Gestures:

Enter a new gesture name in the text field and click Add Gesture.
The gesture is added to the dropdown, and a new directory is created for data collection.


Toggle Theme:

Click Toggle Theme to switch between light and dark modes for a personalized experience.


Quit:

Click Quit to close the application.



🖼️ UI Highlights
The PyQt5-based interface is designed for immersion and ease of use:

Live View Panel: Displays real-time webcam feed with overlaid hand landmarks, scaled to 480x360 for clarity.
Prediction Panel: Shows dynamic updates for detected gestures, confidence, sample count, and FPS.
Button Panel: Features colorful, hover-enabled buttons and a dropdown for gesture selection, styled with modern aesthetics.
Theme Support: Switch between light and dark themes to suit your preference, with smooth transitions for widgets and backgrounds.

📂 Project Structure
hand-gesture-recognition/
├── gesture_data/              # Directory for storing gesture landmark data
│   ├── open_hand/            # Subdirectory for each gesture
│   ├── closed_fist/
│   └── ...
├── gesture_model.pkl          # Trained Random Forest model
├── selector.pkl               # Feature selector for dimensionality reduction
├── gesture_recognition.py     # Main script
├── README.md                  # This file
└── requirements.txt           # Dependency list

🔍 How It Works

Hand Tracking:

MediaPipe detects hand landmarks (21 per hand, with x, y, z coordinates) from webcam frames.
Landmarks are normalized relative to the wrist for position-invariant features.


Data Collection:

Landmarks are saved as .npy files in gesture_data/<gesture_name>.
Data augmentation applies random rotations to enhance model generalization.


Model Training:

Features are selected using SelectKBest (top 30 features).
A Random Forest classifier is trained on normalized and augmented data.


Real-Time Testing:

The model predicts gestures from live webcam input.
A buffer tracks recent gestures to detect predefined sequences.
FPS and confidence are calculated and displayed.


GUI:

Built with PyQt5, using QMainWindow for the main layout.
Threading ensures responsive operation during data collection and testing.
Styled with CSS-like properties for a modern, immersive look.



🛠️ Troubleshooting

Webcam Issues:
Ensure your webcam is connected and accessible.
Check if cv2.VideoCapture(0) works; try other indices (e.g., 1) if needed.


Missing Model/Selector:
Train the model first if gesture_model.pkl or selector.pkl is missing.


GUI Freezing:
Threading is used to prevent freezing, but ensure sufficient CPU resources.


Dependency Errors:
Verify all dependencies are installed correctly.
Use a virtual environment to avoid conflicts:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt





🌟 Future Enhancements

Add model evaluation metrics (e.g., accuracy, confusion matrix).
Support more complex gesture sequences via GUI input.
Optimize performance for lower-end devices.
Add support for multiple cameras or resolutions.
Implement advanced data augmentation (e.g., scaling, translation).

🤝 Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/YourFeature).
Commit your changes (git commit -m 'Add YourFeature').
Push to the branch (git push origin feature/YourFeature).
Open a pull request.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
🙌 Acknowledgments

MediaPipe: For robust hand tracking.
scikit-learn: For efficient machine learning tools.
PyQt5: For a powerful and flexible GUI framework.
OpenCV: For seamless video capture and processing.


Ready to control your world with a wave of your hand? Dive into the Hand Gesture Recognition System and experience the power of computer vision with an immersive UI! For questions or support, open an issue or contact the project maintainers.
