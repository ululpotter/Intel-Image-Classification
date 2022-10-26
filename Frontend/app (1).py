import streamlit as st
import numpy as np
from skimage import io
from skimage.transform import resize
from PIL import Image
import tensorflow as tf
import cv2
from keras_preprocessing import image

#Tambahkan File Haarcascade Frontal Face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#masukan inputan gambar
with st.form(key='form_parameters'):
    st.title("Emotion Detection")
    uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'png', 'jpeg'])
    
    st.markdown('---')
    submitted = st.form_submit_button('Predict')


#load model
model = tf.keras.models.model_from_json(open('model.json').read())
model.load_weights('seq.h5')

#masukan jenis emosi
emotions = ('Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut','Biasa')


if submitted:
    img = np.array(Image.open(uploaded_file))
    # img = Image.open(uploaded_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
         detected_face = img[int(y):int(y + h), int(x):int(x + w)]
         detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) 
         detected_face = cv2.resize(detected_face, (48, 48))
         img_pixels = image.img_to_array(detected_face)
         img_pixels = np.expand_dims(img_pixels, axis=0)
         img_pixels /= 255
         predictions = model.predict(img_pixels)
         max_index = np.argmax(predictions[0])
         emotion = emotions[max_index]
         cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
         cv2.imshow('img', img)
         st.title(emotion)
         st.image(img)