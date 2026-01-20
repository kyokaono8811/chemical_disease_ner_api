from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

# Create the FastAPI app
app = FastAPI(
    title="Chemical/Disease NER BERT Model API",
    description="Chemical/Disease NER using fine-tuned LoRA BERT",
    version="1.0.0"
)

# Global variables - will hold the model and tokenizer
model = None
tokenizer = None
device = None


class TextInput(BaseModel):
    text: str


class Entity(BaseModel):
    word: str
    entity: str
    score: float


class NERResponse(BaseModel):
    text: str
    entities: List[Entity]


@app.on_event("startup")
async def load_model():

    global model, tokenizer, device
    
    try:
        print("=" * 60)
        print("Loading your trained model...")
        print("=" * 60)
        
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" Using device: {device}")
        
        # Define the label mappings (same as training)
        id_to_label = {
            0: 'O',
            1: 'B-Chemical',
            2: 'B-Disease',
            3: 'I-Chemical',
            4: 'I-Disease'
        }
        label_to_id = {label: id for id, label in id_to_label.items()}
        
        # Load the LoRA configuration
        peft_config = PeftConfig.from_pretrained("./model")
        print(f" Loaded LoRA config")
        
        # Load the base BERT model with label mappings
        base_model = AutoModelForTokenClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=5,
            id2label=id_to_label,
            label2id=label_to_id
        )
        print(f" Loaded base model: {peft_config.base_model_name_or_path}")
        
        # Load your fine-tuned LoRA weights on top
        model = PeftModel.from_pretrained(base_model, "./model")
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        print(f" Loaded LoRA adapter")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("./model")
        print(f" Loaded tokenizer")
        
        print("=" * 60)
        print(" Model loaded successfully")
        print("=" * 60)
        
    except Exception as e:
        print(f" Error loading model: {str(e)}")
        raise e


@app.get("/")
async def root():
    return {
        "message": "API is running",
        "status": "healthy",
        "docs": "Visit /docs for interactive API documentation"
    }


@app.get("/health")
async def health_check():

    return {
        "status": "healthy" if model is not None else "model not loaded",
        "device": str(device),
        "model_ready": model is not None,
        "tokenizer_ready": tokenizer is not None
    }


@app.post("/predict", response_model=NERResponse)
async def predict_entities(input_data: TextInput):

    # Check if model is loaded
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded yet. Wait a moment and try again."
        )
    
    try:
        # Tokenize the input text
        inputs = tokenizer(
            input_data.text,
            return_tensors="pt",
            truncation=True,
            padding="max_length,
            max_length=128
        ).to(device)
        
        # Get predictions from the model
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
            scores = torch.softmax(outputs.logits, dim=2)
        
        # Convert predictions to readable format
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predictions = predictions[0].cpu().numpy()
        scores = scores[0].cpu().numpy()
        
        # Group tokens into words and extract entities
        entities = []
        current_entity = None
        
        for idx, (token, pred, score) in enumerate(zip(tokens, predictions, scores)):
            # Skip special tokens
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            # Get label and confidence
            label = model.config.id2label.get(pred, f"LABEL_{pred}")
            confidence = float(score[pred])
            
            # Handle subword tokens (##)
            if token.startswith("##"):
                if current_entity:
                    current_entity["word"] += token[2:]
            else:
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)
                
                # Start new entity (if not "O" = Outside)
                if label != "O":
                    current_entity = {
                        "word": token,
                        "entity": label,
                        "score": confidence
                    }
                else:
                    current_entity = None
        
        # Add last entity
        if current_entity:
            entities.append(current_entity)
        
        return NERResponse(
            text=input_data.text,
            entities=entities
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error during prediction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
