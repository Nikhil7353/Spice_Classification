from django.shortcuts import render
 # Install pillow instead of PIL
import numpy as np
import PIL
from PIL import Image, ImageOps
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops


# Function to compute color histogram
def compute_color_histogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bins = 8
    hue_hist = np.histogram(hsv_image[:,:,0], bins=bins, range=(0, 180))[0]
    saturation_hist = np.histogram(hsv_image[:,:,1], bins=bins, range=(0, 256))[0]
    value_hist = np.histogram(hsv_image[:,:,2], bins=bins, range=(0, 256))[0]
    color_histogram = np.concatenate((hue_hist, saturation_hist, value_hist))
    return color_histogram


def compute_texture_features(image):
    glcm = graycomatrix(image, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    texture_features = np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation])
    return texture_features

def load_trained_models():
    # with open('home/knn_classifier.pkl', 'rb') as f:
    #     knn_classifier = pickle.load(f)
    with open('home/svm_classifier.pkl', 'rb') as f:
        svm_classifier = pickle.load(f)

    return svm_classifier

def test_single_image(image_path, svm_classifier, class_names):
    # Load the image
    image = cv2.imread(image_path)

    # Extract features from the image
    color_histogram = compute_color_histogram(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_feature = compute_texture_features(gray_image)

    # Combine features (excluding shape features)
    combined_feature = np.concatenate([color_histogram, texture_feature]).reshape(1, -1)

    svm_prediction = class_names[int(svm_classifier.predict(combined_feature)[0])]

    return svm_prediction


import os
from django.conf import settings

def index(request):
    return render(request,'index2.html')

def start_detection(request):
    return render(request,'quality_detection.html')

def get_results(request):
    if request.method == 'POST':
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = request.FILES['image']
        print(image)
        image_path = os.path.join(settings.MEDIA_ROOT, image.name)
        with open(image_path, 'wb') as f:
            for chunk in image.chunks():
                f.write(chunk)
        

        class_names = ["Cinnamon", "Black pepper", "Clove", "Green Cardamon","Coriander"]

        # image = Image.open(image).convert("RGB")
        # size = (224, 224)
        # image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        # image_array = np.asarray(image)
        # normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        # data[0] = normalized_image_array
        # type_prediction = type_model.predict(data)
        # type = np.argmax(type_prediction)
        # quality_prediction = quality_model.predict(data)
        # quality = np.argmax(quality_prediction)
        svm_classifier = load_trained_models()
        result = test_single_image(image_path, svm_classifier, class_names)
        print(result)
        if result == "Cinnamon":
            description = "Cinnamon, a spice with a sweet and warm flavor, is known for its rich nutritional profile and numerous health benefits. It contains antioxidants and anti-inflammatory properties, which may protect against cell damage and reduce inflammation. Cinnamon is also linked to improved heart health, blood sugar control, and a reduced risk of neurodegenerative diseases. Additionally, it is a good source of manganese, fiber, and calcium."

        elif result == "Black pepper":
            description = "Black pepper, known for its bold flavor, is not just a spice; it's also a source of antioxidants like piperine, offering cell protection and anti-inflammatory properties. Beyond its taste, black pepper aids digestion by stimulating enzymes and can enhance nutrient absorption. Low in calories yet rich in vitamin K, vitamin C, iron, and calcium, black pepper adds more than just taste to your dishes—it adds a health boost too."
        
        elif result == "Clove":
            description = "Clove, a spice with a strong, sweet, and aromatic flavor, is prized for its many health benefits and rich nutritional content. It is packed with antioxidants and has anti-inflammatory properties, which may help protect against cell damage and reduce inflammation in the body. Clove has been used traditionally to alleviate digestive issues, improve liver health, and even aid in controlling blood sugar levels. Additionally, it is a good source of vitamins and minerals such as vitamin C, vitamin K, and manganese."

        elif result == "Green Cardamon":
            description = "Green cardamom, known for its strong, aromatic flavor, is a spice celebrated for its health benefits and nutritional value. It is rich in antioxidants and has anti-inflammatory properties, which may help protect cells from damage and reduce inflammation. Green cardamom is also believed to aid digestion, improve oral health, and even have potential cancer-fighting properties. Additionally, it is a good source of minerals like potassium, calcium, and magnesium, as well as vitamins A and C."

        elif result == "Coriander":
            description = "Coriander, also known as cilantro or Chinese parsley, is an herb with a bright, citrusy flavor used in cuisines worldwide. It is rich in antioxidants and has anti-inflammatory properties, which may help protect against cell damage and reduce inflammation. Coriander is also believed to aid digestion, improve cholesterol levels, and promote skin health. Additionally, it is a good source of vitamins A, C, and K, as well as minerals like manganese, iron, and calcium."

        else:
            description = ""
        return render(request,'get_results.html',{'result': result,'description': description})
    return render(request,'quality_detection.html')