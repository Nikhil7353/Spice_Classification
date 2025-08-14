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
  - git clone https://github.com/yourusername/spice-identification.git
  - cd spice-identification

### 2️⃣ Create Virtual Environment & Install Dependencies
  - python -m venv env
  - source env/bin/activate   # Linux/Mac
  - env\Scripts\activate      # Windows

  pip install -r requirements.txt

3️⃣ Run the Application
  python manage.py runserver
Open your browser and go to: http://127.0.0.1:8000/

📊 Model Development Process

1. Data Collection
    Custom dataset created with DSLR camera.
    5 spices: Clove, Green Cardamom, Cinnamon, Black Pepper, Coriander.
    40 original images per class, multiple angles, white background.

2. Data Preprocessing
    Resizing and normalization.
    Noise reduction.    
    Data augmentation using ImageDataGenerator:
    Width & height shifts (±10%)
    Zoom (±10%)
    No flipping to preserve natural orientation.

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

📈 Experimental Results
| Algorithm     | Accuracy (%) |
| ------------- | ------------ |
| Naive Bayes   | 88.00        |
| Decision Tree | 88.00        |
| kNN           | 92.00        |
| Random Forest | 92.00        |
| **SVM**       | **96.00**    |

📷 Screenshots
1. Home Page

2. Image Upload Interface

3. Uploaded Image Example

4. Detection Result with Nutritional Benefits

🚀 Future Enhancements
    Expand dataset with more spice categories.
    Use Convolutional Neural Networks (CNNs) for even higher accuracy.
    Develop mobile application for on-the-go detection.
    Implement real-time camera-based detection.
    Add global spice varieties beyond Indian spices.
