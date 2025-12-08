🤖 **Real-Time Sequential Sign Recognition and Translation**
A desktop application built with PySide6 (Qt) that uses Machine Learning (Logistic Regression) and MediaPipe to detect sign language gestures in real-time, converts them into a sentence, and provides text-to-speech (TTS) and translation services.

✨ **Features**
Real-Time Gesture Detection: Utilizes the webcam to capture hand landmarks via MediaPipe Hands.

Sequential Recognition: Stabilizes predictions using a frame buffer and builds detected signs into a continuous sentence.

Intuitive UI/UX: Features a dark, dynamic interface built with PySide6 that automatically scales to the window size.

Multilingual Translation: Translates the recognized English sentence into dozens of international and Indian languages using Google Translate API.

Text-to-Speech (TTS): Converts the final (English or translated) sentence into speech using gTTS and plays it back using a thread-safe Pygame Mixer implementation.

🛠️ **Project Structure**
The project is organized to separate application logic, training scripts, and data:

RealTime_Sign_Recognition/
├── app_pyside.py             # MAIN GUI APPLICATION (PySide6)
├── .gitignore                # Ignores virtual environment, cache, and large files
├── **1_data_collection.py**  # Script to capture data from webcam
├── **2_train_model.py**      # Script to train the ML model
├── **3_app.py**              # Console-based detection script (Non-GUI version)
├── **data/**                 # Stores collected sign language features (must be created)
│   └── sign_language_data.csv   # Collected landmark data
└── **model/**                # Stores the trained ML model (must be created)
    └── sign_language_model.p  # The trained Logistic Regression classifier
    
    
🚀**1.Getting Started**
Follow these steps to set up the environment, collect data, train the model, and run the GUI application.
____________________________________________________________________________________
1. Setup and Dependencies
Clone the repository:

git clone https://github.com/YOUR_USERNAME/RealTime-Sign-Recognition.git
cd RealTime-Sign-Recognition

____________________________________________________________________________________
Create Folders: You must manually create the empty directories before running the scripts:

mkdir data
mkdir model
____________________________________________________________________________________
Setup and Activate Virtual Environment:

python -m venv venv-mp-signs
# Activate the environment (Use the command appropriate for your OS/Shell)
____________________________________________________________________________________
Install Required Libraries:

pip install -r requirements.txt
____________________________________________________________________________________


**2. Data Collection**
Run the 1_data_collection.py script to populate the data/sign_language_data.csv file. Repeat this process for every unique sign you want the application to recognize.

python 1_data_collection.py

# Follow the prompts to enter labels and press 'C' to capture samples.
____________________________________________________________________________________

**3. Model Training**
After collecting sufficient data, train the model. This script will automatically create and save the required model/sign_language_model.p file.

python 2_train_model.py
____________________________________________________________________________________

**4. Running the Final Application**
With the model file in place, you can launch the GUI.

python app_pyside.py

The application will launch. Click START CAMERA, perform your trained signs, and use the Translate and Text to Voice functions.
____________________________________________________________________________________

🤝 Contribution
Contributions are welcome! If you have suggestions for new features, bug fixes, or performance improvements, please open an issue or submit a pull request.
