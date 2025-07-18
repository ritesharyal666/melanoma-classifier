import os
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Model loading with error handling
MODEL_PATH = "app/models/final.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
    logger.info(f"Input shape: {model.input_shape}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError("Could not load ML model") from e

def preprocess_image(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    """Preprocess uploaded image for model prediction"""
    try:
        # Open and resize image
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(target_size)
        
        # Convert to array and normalize
        image_array = np.array(image) / 255.0
        
        # Add batch dimension
        return np.expand_dims(image_array, axis=0)
    except UnidentifiedImageError:
        raise ValueError("Invalid image file")
    except Exception as e:
        raise RuntimeError(f"Image processing failed: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                {"error": "Only image files are allowed"},
                status_code=400
            )
        
        # Read and process image
        contents = await file.read()
        processed_image = preprocess_image(contents)
        
        # Make prediction
        prediction = model.predict(processed_image)
        confidence = float(prediction[0][0])
        result = "malignant" if confidence > 0.5 else "benign"
        
        return JSONResponse({
            "result": result,
            "confidence": float(confidence if result == "malignant" else 1 - confidence),
            "message": "Prediction successful"
        })
        
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return JSONResponse(
            {"error": "An error occurred during prediction"},
            status_code=500
        )