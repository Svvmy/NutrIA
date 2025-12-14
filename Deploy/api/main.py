import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import json
import os

# Initialize FastAPI
app = FastAPI(
    title="Food-101 Prediction API",
    description="API for classifying food images using a MobileNetV2 model.",
    version="1.0.0"
)

# Global variables
model = None
class_names = None

# Standard Food-101 Class Names
FOOD101_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
    'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
    'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
    'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
    'waffles'
]

@app.on_event("startup")
def load_model():
    global model, class_names
    model_path = "../model/mobilenetv2_v3_FullFT_Regularized.h5"
    
    # Load Model
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: Model file not found at {model_path}")

    # Use standard classes
    class_names = FOOD101_CLASSES

@app.get("/")
def home():
    return {"message": "Welcome to the Food-101 Prediction API. POST an image to /predict."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize to 224x224 (Standard for MobileNetV2)
        image = image.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Expand dimensions (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess input (MobileNetV2 specific: scales to [-1, 1])
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Predict
        predictions = model.predict(img_array)
        
        # Get top 5 predictions
        top_k = 5
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        top_predictions = []
        if class_names:
            for i in top_indices:
                top_predictions.append({
                    "class": class_names[i],
                    "probability": float(predictions[0][i])
                })
        
        # Get top prediction for backward compatibility/easy access
        predicted_index = top_indices[0]
        predicted_class = class_names[predicted_index] if class_names else "Unknown"
        probability = float(predictions[0][predicted_index])
            
        return {
            "class": predicted_class,
            "class_index": int(predicted_index),
            "probability": probability,
            "top_predictions": top_predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
