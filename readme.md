# 🧠 Brain Tumor Detection 

## 📌 Overview

Brain Tumor Detection is a Machine Learning and Computer Vision project designed to classify MRI brain scan images and detect the presence of tumors. The system utilizes deep learning techniques to analyze medical images and provide quick predictions through a user-friendly web interface.

The project consists of a trained deep learning model, a Python-based backend for inference, and a responsive frontend that allows users to upload MRI images and receive instant predictions.

This application demonstrates the use of Artificial Intelligence in healthcare by assisting in the early detection of brain tumors from MRI scans.

---

## 🚀 Features

* Brain MRI Image Classification
* Tumor Detection Using Deep Learning
* Image Upload Functionality
* Real-Time Prediction Results
* User-Friendly Web Interface
* Model Training and Evaluation Pipeline
* Fast and Accurate Image Processing
* Responsive Frontend Design

---

## 🛠️ Technologies Used

### Programming Languages

* Python
* JavaScript
* HTML5
* CSS3

### Machine Learning & Deep Learning

* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-Learn

### Backend

* Flask

### Frontend

* HTML
* CSS
* JavaScript

---

## 📂 Project Structure

```text
brain-tumor-detection/
│
├── backend/
│   ├── training/
│   │   └── train.py          # Model training script
│   ├── app.py               # Flask backend application
│   └── requirements.txt     # Backend dependencies
│
├── frontend/
│   ├── index.html           # Main user interface
│   ├── style.css            # Styling
│   └── script.js            # Frontend functionality
│
├── README.md
```

---

## 🎯 Project Objective

The objective of this project is to develop an AI-powered system capable of identifying brain tumors from MRI images, helping support medical diagnosis through automated image analysis.

---

## ⚙️ System Workflow

### Step 1: Image Upload

Users upload a brain MRI image through the web interface.

### Step 2: Image Preprocessing

The image is resized, normalized, and prepared for model prediction.

### Step 3: Deep Learning Prediction

The trained model analyzes the MRI scan and predicts whether a tumor is present.

### Step 4: Result Display

The prediction result is displayed on the web application.

---

## 🧠 Machine Learning Pipeline

### Data Collection

MRI brain scan images are collected and organized into training and testing datasets.

### Data Preprocessing

* Image resizing
* Normalization
* Dataset splitting
* Data augmentation

### Model Training

The deep learning model is trained using labeled MRI images to learn tumor-related patterns.

### Model Evaluation

Performance is evaluated using classification metrics and validation datasets.

---

## 📊 Key Functionalities

### MRI Image Upload

Users can upload MRI scans directly from their device.

### Tumor Detection

The model predicts whether a tumor is present based on image features.

### Real-Time Results

Predictions are generated instantly through the Flask backend.

### Interactive Interface

A clean frontend provides an easy-to-use experience.

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/veeramalla-manikanta/brain-tumor-detection.git
cd brain-tumor-detection
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r backend/requirements.txt
```

---

## ▶️ Run the Backend

Navigate to the backend folder:

```bash
cd backend
```

Run Flask application:

```bash
python app.py
```

---

## 🌐 Run the Frontend

Open:

```text
frontend/index.html
```

in your browser.

Alternatively, connect the frontend to the running Flask backend.

---

## 🏥 Applications

* Medical Image Analysis
* Healthcare Diagnostics
* MRI Scan Classification
* Clinical Decision Support Systems
* AI-Assisted Healthcare Solutions
* Medical Research Projects

---

## 📈 Future Enhancements

* Multi-Class Tumor Classification
* Tumor Segmentation
* Tumor Localization Using Bounding Boxes
* Explainable AI (XAI) Visualizations
* Cloud Deployment
* Doctor Dashboard Integration
* Mobile Application Support
* Improved Model Accuracy with CNN Architectures

---

## 📊 Project Highlights

* Deep Learning-Based Medical Image Analysis
* End-to-End Web Application
* Automated Brain Tumor Detection
* Real-Time Prediction System
* Healthcare AI Application
* Full-Stack Machine Learning Project

---

## 👨‍💻 Author

### Manikanta Veeramalla

Data Analyst | Machine Learning Enthusiast | Power BI Developer

GitHub:
https://github.com/veeramalla-manikanta

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

Contributions, suggestions, and feedback are always welcome.

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis.

---

## 📜 License

This project is licensed under the MIT License.
