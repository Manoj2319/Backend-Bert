from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize FastAPI
app = FastAPI()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-try-1")
model = AutoModelForSequenceClassification.from_pretrained("bert-try-1")

# Define input and output data structure
class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input_text: InputText):
    try:
        # Tokenize input
        inputs = tokenizer(input_text.text, return_tensors="pt", truncation=True, padding=True)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities (softmax)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities).item()
        
        return {
            "predicted_class": predicted_class,
            "confidence": probabilities.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
