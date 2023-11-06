import streamlit as st
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, array_to_img
from keras.utils import to_categorical
import cv2
from keras.models import load_model

# Load the pre-trained model
model = load_model('ONLYrotate_num_train_changed_32_batch_exported_model_git.h5')

# Prevent openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# Dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Create a Streamlit window
st.title("Emotion Detection 😁🥲🥰😖🥺😡")
picture = st.camera_input("Take a picture")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Create a Streamlit loop to display the webcam feed
if uploaded_image is not None or picture is not None:
    # Read the uploaded image
    if uploaded_image is not None:
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
    else:
        image = cv2.imdecode(np.fromstring(picture.read(), np.uint8), 1)
    
    # Find haar cascade to draw bounding box around face
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No faces found in the image.")
    else:
        st.success(f"{len(faces)} face(s) found")        
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(image, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the processed image in Streamlit
    st.image(image, channels="BGR", use_column_width=True)
