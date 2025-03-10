# Face Recognition System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-red?logo=opencv&logoColor=white)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-orange)

This is a Face Recognition System built using OpenCV, LBPH Face Recognizer, and Tkinter for GUI interaction. The system consists of three main components:

- **Face Capture**: Captures images of individuals and assigns them unique IDs.
- **Model Training**: Trains an LBPH model on the collected face images.
- **Face Recognition**: Recognizes faces using the trained model with a GUI-based interface.

---
## 📁 Project Structure
```
📦 FaceRecognitionSystem
 ┣ 📂 Classifiers
 ┃ ┣ 📜 haarface.xml (Haar Cascade for face detection)
 ┃ ┣ 📜 TrainedLBPH.yml (Trained LBPH model, generated after training)
 ┣ 📂 faces (Folder containing captured images for training)
 ┣ 📜 capture_faces.py (Captures and stores images for training)
 ┣ 📜 train_model.py (Trains LBPH model on stored images)
 ┣ 📜 recognize_faces.py (GUI application for real-time recognition)
 ┣ 📜 id-names.csv (CSV file storing user ID and names)
```

---
## 🚀 Installation
### Prerequisites
Make sure you have the following installed:
- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- Pandas
- Tkinter (Included with Python)

### Install Dependencies
Run the following command to install required libraries:
```bash
pip install opencv-python numpy pandas
```

---
## 🏗 Usage
### 1️⃣ Capture Faces
Run the `capture_faces.py` script to start collecting images for face recognition.
```bash
python capture_faces.py
```
- Enter a unique ID when prompted.
- Position your face within the red rectangle and press `s` to capture images.
- Capture at least **30 images** for better accuracy.
- Press `q` to exit.

### 2️⃣ Train the Model
Once faces are captured, train the model by running:
```bash
python train_model.py
```
- This will create `TrainedLBPH.yml` inside the `Classifiers` folder.

### 3️⃣ Run Face Recognition
To recognize faces in real-time, run:
```bash
python recognize_faces.py
```
- Load the `id-names.csv` file when prompted.
- Click **Start Recognition** to detect faces.
- Recognized names will be displayed along with accuracy percentage.

---
## 🎯 Features
✅ **Real-time Face Recognition** using OpenCV and LBPH.

✅ **GUI Interface** for easy interaction.

✅ **Image Capture** with different angles and lighting conditions.

✅ **Customizable Model** for training with new data.

---
## 🛠 Technologies Used
- ![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
- ![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-red?logo=opencv&logoColor=white)
- ![Tkinter](https://img.shields.io/badge/Tkinter-GUI-orange)
- ![NumPy](https://img.shields.io/badge/NumPy-Matrix-blue?logo=numpy&logoColor=white)
- ![Pandas](https://img.shields.io/badge/Pandas-Dataframe-blue?logo=pandas&logoColor=white)

---
## 🤝 Contributing
Feel free to contribute by improving the model, enhancing the GUI, or optimizing the recognition system. 

---
## 📩 Contact
📧 Email: [m.montasser.elazab@gmail.com](mailto:m.montasser.elazab@gmail.com)  
🔗 LinkedIn: [Mohamed Montasser](https://www.linkedin.com/in/mohamed-montasser-/)

---
## 📝 License
This project is open-source and free to use under the MIT License.
