# 🌶️ Identification and Classification of Indian Spices

## 📌 Project Overview
**Identification and Classification of Indian Spices** is a machine learning-based web application that automatically detects and classifies Indian spices from images.  
It uses **image processing** and **multiple ML algorithms** to deliver accurate predictions, along with **nutritional benefits** of the identified spice.

This project aims to help:
- Culinary enthusiasts
- Professional chefs
- Food industry professionals
- Researchers
- Healthcare and wellness experts  
…by providing **consistent, fast, and reliable spice recognition**.

---

## ✨ Features
- Upload an image and get **instant spice classification**.
- Supports **five spice classes**:
  - Clove
  - Green Cardamom
  - Cinnamon
  - Black Pepper
  - Coriander
- Displays **nutritional and health benefits** for each spice.
- Implements **five ML algorithms**:
  - Naive Bayes (NB)
  - Decision Tree (DT)
  - k-Nearest Neighbors (kNN)
  - Random Forest (RF)
  - Support Vector Machine (SVM)
- **SVM** achieved the highest accuracy.
- Dataset augmented for better generalization.
- Scalable for adding more spice categories.

---

## 🛠 Tech Stack
**Frontend:**
- HTML5, CSS3, JavaScript(Web-based UI)

**Backend:**
- Python, Django

**Machine Learning:**
- scikit-learn
- NumPy, Pandas
- TensorFlow (optional for deep learning expansion)

**Image Processing:**
- OpenCV
- Pillow (PIL)

**Database:**
- SQLite / PostgreSQL
-  Custom-captured spice images (40 images per class, augmented)

**Version Control:**
- Git & GitHub

---

## 📂 Project Structure
- Spice-Identification/
- │── dataset/ # Original and augmented spice images
- │── models/ # Trained ML model files
- │── src/
- │ ├── data_preprocessing.py
- │ ├── feature_extraction.py
- │ ├── model_training.py
- │ ├── prediction.py
- │── webapp/
- │ ├── templates/ # HTML templates
- │ ├── static/ # CSS, JS, Images
- │ ├── views.py # Django views
- │ ├── urls.py # URL routing
- │── requirements.txt
- │── README.md
  
---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
    bash
  - git clone https://github.com/Nikhil7353/Spice_Classification
  - cd spice-identification

### 2️⃣ Create Virtual Environment & Install Dependencies
  - python -m venv env
  - source env/bin/activate   # Linux/Mac
  - env\Scripts\activate      # Windows

   pip install -r requirements.txt

### 3️⃣ Run the Application
 - python manage.py runserver
 - Open your browser and go to: http://127.0.0.1:8000/

## 📊 Model Development Process

  1. Data Collection
  - Custom dataset created with DSLR camera.
  - 5 spices: Clove, Green Cardamom, Cinnamon, Black Pepper, Coriander.
  - 40 original images per class, multiple angles, white background.

  2. Data Preprocessing
  - Resizing and normalization.
  - Noise reduction.    
  - Data augmentation using ImageDataGenerator:
  - Width & height shifts (±10%)
  - Zoom (±10%)
  - No flipping to preserve natural orientation.

  3. Feature Extraction
    Color histograms
    Texture features via GLCM (contrast, homogeneity, etc.)
    Shape features: area, perimeter, circularity.

  4. Algorithms Implemented
    Naive Bayes
    Decision Tree    
    k-Nearest Neighbors
    Random Forest
    Support Vector Machine

  5. Model Evaluation
    80% training / 20% testing split.
    Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix.
    

---

## 📈 Experimental Results
| Algorithm     | Accuracy (%) |
| ------------- | ------------ |
| Naive Bayes   | 88.00        |
| Decision Tree | 88.00        |
| kNN           | 92.00        |
| Random Forest | 92.00        |
| **SVM**       | **96.00**    |

## 📷 Screenshots
1. Home Page

2. Image Upload Interface

3. Uploaded Image Example

4. Detection Result with Nutritional Benefits

## 🚀 Future Enhancements
 - Expand dataset with more spice categories.
 - Use Convolutional Neural Networks (CNNs) for even higher accuracy.
 - Develop mobile application for on-the-go detection.
 - Implement real-time camera-based detection.
 - Add global spice varieties beyond Indian spices.

## 📄 License

This project is open source under the MIT License. You are free to use, modify, and distribute it.

## 🔗 Contribution
Contributions, issues, and feature requests are welcome!
Feel free to fork the repository and submit pull requests.

## 📸 Screenshot

### 🔧 1. Django Server Running in VS Code
The development server is successfully launched using Django (python manage.py runserver). The system checks completed with no issues, and routes like /start_detection/ are responding correctly, confirming the backend is functional and listening at http://127.0.0.1:8000.

<img width="1920" height="1080" alt="server_running" src="https://github.com/user-attachments/assets/2639c0b1-dc5b-478a-a143-5ecc1d601f77" />

### 🏠 2. Home Page – Start Detection
This is the application’s home screen, featuring a clean UI with spice images in the background. Users can click the green "Start Spice Detection" button to proceed to the detection interface. The layout is intuitive, ideal for both desktop and mobile views.

<img width="1920" height="1020" alt="home_page" src="https://github.com/user-attachments/assets/016d8f7b-a9cb-41a9-8649-0371cebecc76" />

### 📤 3. Image Upload Interface
The Spices Detection screen allows users to upload an image of an Indian spice. Once an image is uploaded, it is previewed in the center box, and users can click "Detect" to initiate classification. The background remains consistent for brand continuity.

<img width="1920" height="1080" alt="upload_interface" src="https://github.com/user-attachments/assets/edc8db11-e5d2-4f1b-922d-286c169e0ad4" />

### 🖼️ 4. Uploaded Image Preview
An image of Black Pepper has been uploaded and is now visible in the preview area. This step ensures users can verify the correct file before detection. It enhances usability and helps prevent incorrect classification due to wrong image uploads.
This is a representative example. The same upload and preview process applies to all five spice classes: Clove, Green Cardamom, Cinnamon, Black Pepper, and Coriander.

<img width="1920" height="1080" alt="image_uploaded" src="https://github.com/user-attachments/assets/9a6a564d-91e5-4f58-80fb-da959677b173" />

### ✅ 5. Detection Result with Nutritional Info
The detected spice is displayed clearly (Black pepper in this case), along with a detailed explanation of its health and nutritional benefits. This informative output is useful for chefs, researchers, or anyone curious about spice properties and health impacts.

<img width="1920" height="1080" alt="detection_result" src="https://github.com/user-attachments/assets/38a1f744-0d94-4646-9d2d-127f51b2530e" />
