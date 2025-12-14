import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd

# API URL
API_URL = "http://api:8000/predict"  # Use 'api' hostname for Docker, 'localhost' for local

st.set_page_config(page_title="Food-101 Classifier", layout="centered")

st.title("🍔 Food-101 Image Classifier")
st.write("Upload a food image to identify it!")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        with st.spinner("Classifying..."):
            try:
                # Prepare file for API
                # Reset pointer to beginning of file
                uploaded_file.seek(0)
                files = {"file": ("image.jpg", uploaded_file, "image/jpeg")}
                
                # Call API
                # Note: When running locally without docker, change API_URL to localhost
                try:
                    response = requests.post(API_URL, files=files)
                except requests.exceptions.ConnectionError:
                     # Fallback for local run if Docker networking isn't set up yet
                     response = requests.post("http://127.0.0.1:8000/predict", files=files)

                if response.status_code == 200:
                    result = response.json()
                    predicted_class = result["class"]
                    probability = result["probability"]
                    
                    st.success(f"### Prediction: {predicted_class}")
                    st.write(f"Confidence: {probability:.2%}")
                    st.progress(int(probability * 100))
                    
                    # Top 5 Visualization
                    if "top_predictions" in result:
                        st.write("### Top 5 Probabilities")
                        top_preds = result["top_predictions"]
                        
                        # Create DataFrame for Visualization
                        df = pd.DataFrame(top_preds)
                        df = df.set_index("class")
                        
                        # Display Chart
                        st.bar_chart(df["probability"])
                        
                        # Optional: Display as a small table if needed, but chart is good.
                        # st.table(df)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
