from PIL import Image
import numpy as np
from numpy import asarray
import tensorflow as tf
import streamlit as st

class_names = ['Early Blight Detected',
               'Late Blight Detected', 'Healthy Plant']


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
        label='Upload plant leaf image to test : ', type=['png', 'jpg', 'jpeg'])

    if img_data:
        st.write("Filename: ", img_data.name)
        st.write("File type: ", img_data.type)
        # display image
        uploaded_image = Image.open(img_data)
        st.image(uploaded_image)

        resized_img = uploaded_image.resize((256, 256))
        numpydata = asarray(resized_img)

        predicted_class, confidence = predict(model, numpydata)
        st.write("Prediction : ", predicted_class)
        st.write("Prediction Confidence : ", confidence)
