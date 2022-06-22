from PIL import Image
import numpy as np
from numpy import asarray
import tensorflow as tf
import streamlit as st

class_names = ['Potato___Early_blight',
               'Potato___Late_blight', 'Potato___healthy']

# img_path = 'training/PlantVillage/Potato___healthy/0b3e5032-8ae8-49ac-8157-a1cac3df01dd___RS_HL 1817.JPG'


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


if __name__ == '__main__':
    model = tf.keras.models.load_model('potatoes.h5')

    img_data = st.file_uploader(
        label='Upload an image to test', type=['png', 'jpg', 'jpeg'])

    if img_data:
        st.write("Filename: ", img_data.name)
        st.write("File type: ", img_data.type)
        # display image
        uploaded_image = Image.open(img_data)
        st.image(uploaded_image)

        # img = Image.open(uploaded_image)
        resized_img = uploaded_image.resize((256, 256))
        numpydata = asarray(resized_img)

        predicted_class, confidence = predict(model, numpydata)
        # print(predicted_class, confidence)
        st.write("Predicted Class : ", predicted_class)
        st.write("Confidence : ", confidence)
