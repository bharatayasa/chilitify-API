import os
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

model_path = './model/model.h5'

model = load_model(model_path)

def load_and_preprocess_image(image: UploadFile, target_size=(224, 224)):
    image_bytes = BytesIO(image.file.read())
    img = load_img(image_bytes, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0 
    return img_array

def predict_image(image: UploadFile, model):
    img_array = load_and_preprocess_image(image)
    prediction = model.predict(img_array)
    class_names = ['Healthy', 'Leaf Curl', 'Leaf Spot', 'Whitefly', 'Yellowish']
    description = ['Healthy dsc', 'Leaf Curl dsc', 'Leaf Spot dsc', 'Whitefly dsc', 'Yellowish dsc']
    prevention = ['Healthy prev', 'Leaf Curl prev', 'Leaf Spot prev', 'Whitefly prev', 'Yellowish prev']
    predicted_class = class_names[np.argmax(prediction)]
    predicted_description = description[np.argmax(prediction)]
    predicted_prevention = prevention[np.argmax(prediction)]
    confidence = float(np.max(prediction))  # Convert to native float
    return predicted_class, predicted_description, predicted_prevention, confidence

app = FastAPI()

@app.get("/")
def index():
    return {"message": "Hello world from ML endpoint!"}

@app.post("/predict_image/")
async def predict_image_endpoint(file: UploadFile = File(...)):
    try:
        predicted_class, predicted_description, predicted_prevention, confidence = predict_image(file, model)
        return {
            "class": predicted_class,
            "description": predicted_description,
            "prevention": predicted_prevention,
            "confidence": confidence,
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Listening to http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
