from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize FastAPI
app = FastAPI()

# Load the tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained("bert-try-1")
    model = AutoModelForSequenceClassification.from_pretrained("bert-try-1")
except Exception as e:
    raise RuntimeError(f"Error loading model or tokenizer: {e}")

# Define input data structure
class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input_text: InputText):
    try:
        # Tokenize input text
        inputs = tokenizer(
            input_text.text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512  # Ensure maximum token length for BERT compatibility
        )
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Get the predicted class
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        return {
            "predicted_class": predicted_class,
            "confidence": probabilities[0].tolist()  # Ensure confidence scores are properly serialized
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# Enable CORS for frontend requests
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://front-end-gamma-flax.vercel.app", "https://front-end-gamma-flax.vercel.app/predict"],  # Replace with your frontend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
