# app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
import io
import joblib
from app.utils import preprocess_image

app = FastAPI()

# Load trained model
model = joblib.load("app/model.pkl")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")  # Grayscale
        image_array = preprocess_image(image)
        prediction = model.predict([image_array])[0]
        return JSONResponse(content={"prediction": int(prediction)})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
