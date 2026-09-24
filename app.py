import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Configuration & Title
st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")
st.title("🔢 MNIST Handwritten Digit Classifier")
st.write("Apni handwritten digit (0-9) ki image upload karen aur model predict karega.")

# 2. Model Load Karne ka Function
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('mnist.keras')

try:
    model = load_my_model()
    st.sidebar.success("Model successfully load ho gaya!")
except Exception as e:
    st.sidebar.error(f"Model load karne mein masala hai: {e}")

# 3. Custom Labels (Aap ki notebook ka ifi function)
def get_custom_label(output):
    labels = {
        0: "ya apka result hai ",
        1: "ya 1 hai",
        2: "ya two hai",
        3: "ya 3 hai",
        4: "this ia four",
        5: "this is five",
        6: "this is 6",
        7: "ya 7 hai",
        8: "8",
        9: "nine"
    }
    return labels.get(output, "Unknown")

# 4. File Uploader UI
uploaded_file = st.file_uploader("Handwritten digit image upload karen (.png, .jpg, .jpeg)...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=200)
    
    if st.button('Predict Digit'):
        with st.spinner('Model predict kar raha hai...'):
            try:
                # Image Preprocessing (Aap ki notebook ke mutabiq)
                img = image.convert("L")  # Grayscale conversion
                img = img.resize((28, 28)) # MNIST standard size
                img_array = np.array(img)
                img_flatten = img_array.reshape(1, 784) # Flatten to 784 features
                
                # Model Prediction
                pred = model.predict(img_flatten)
                output_digit = np.argmax(pred)
                
                # Result Output
                custom_message = get_custom_label(output_digit)
                st.success(f"**Predicted Digit:** {output_digit}")
                st.info(f"**Custom Message:** {custom_message}")
                
            except Exception as e:
                st.error(f"Prediction ke dauran error aaya: {e}")
