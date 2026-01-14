import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((150, 150))
    img_array = np.array(image)
    img_array = img_array / 255.0
    # Batch-Dimension hinzufügen (Modell erwartet Form (1, 150, 150, 3))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cats_dogs_model.keras")
model = load_model()

st.title("Cats and Dogs Classifier")
st.write("...")
uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Bild anzeigen
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
    # Button Klassifizierung
    if st.button("Klassifizieren"):
        st.write("Verarbeite Bild...")
        # Vor dem Klassifizieren das Bild entsprechend vorverarbeiten
        x = preprocess_image(image)
        # Vorhersage
        pred = model.predict(x)[0][0]
        if pred < 0.5:
            label = "Cat"
        else:
            label = "Dog"
        st.write("Ergebnis:", label)
        st.write("Wahrscheinlichkeit für Hund:", float(pred))
        st.write("Hinweis: Der angezeigte Zahlenwert ist eine Wahrscheinlichkeit. Werte nahe 0.5 bedeuten, dass sich das Modell nicht sicher ist.")