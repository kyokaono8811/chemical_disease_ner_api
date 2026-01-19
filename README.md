# Chemical/Disease Named Entity Recognition API

A production-ready REST API for biomedical Named Entity Recognition (NER) using a fine-tuned BERT model with LoRA (Low-Rank Adaptation).

## Overview

This project deploys a BERT-based NER model for identifying:
- **Chemicals** (e.g., Aspirin, Ibuprofen)
- **Diseases** (e.g., diabetes, headaches)

The model is fine-tuned using LoRA for efficient training and deployment. 

## Features

- FastAPI-based REST API
- Dockerized for easy deployment
- LoRA fine-tuned BERT model
- Trained on BC5CDR biomedical dataset

## Model Performance

- **Base Model:** BERT-base-uncased
- **Fine-tuning Method:** LoRA (r=8, alpha=16)
- **Training Dataset:** BC5CDR (BioCreative V Chemical Disease Relation)
- **Task:** Named Entity Recognition 

## Tech Stack

- **Framework:** FastAPI
- **ML Libraries:** PyTorch, Transformers, PEFT
- **Deployment:** Docker, Docker Compose
- **Model:** BERT with LoRA adapters

## Installation

### Prerequisites

- Docker and Docker Compose
- OR Python 3.10+

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ner-api.git
cd ner-api

# Build and run
docker-compose up --build
```

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ner-api.git
cd ner-api

# Install dependencies
pip install -r requirements.txt

# Run the API
python app.py
```

## Usage

### Interactive API Documentation

Once running, visit: `http://localhost:8000/docs`

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Aspirin is used to treat headaches"}
)

result = response.json()
print(result)
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Diabetes is treated with insulin"}'
```

### Expected Response

```json
{
  "text": "Aspirin is used to treat headaches",
  "entities": [
    {
      "word": "Aspirin",
      "entity": "B-Chemical",
      "score": 0.956
    },
    {
      "word": "headaches",
      "entity": "B-Disease",
      "score": 0.892
    }
  ]
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint (API info) |
| `/health` | GET | Health check |
| `/predict` | POST | Single text prediction |
| `/batch_predict` | POST | Batch predictions |
| `/docs` | GET | Interactive API docs |

## Project Structure

```
.
├── app.py                 
├── requirements.txt       
├── Dockerfile            
├── docker-compose.yml    
├── notebooks/
│   └── finetune_bert.ipynb 
└── README.md             
```
## Configuration

Key parameters in `app.py`:
- `num_labels`: 5 (O, B-Chemical, I-Chemical, B-Disease, I-Disease)
- `max_length`: 512 tokens
- `device`: Auto-detected (CPU/GPU)

## Model Training

The model was trained using:
- **Dataset:** BC5CDR (tner/bc5cdr from Hugging Face)
- **Base Model:** google-bert/bert-base-uncased
- **Method:** LoRA fine-tuning
- **Framework:** Hugging Face Transformers + PEFT

See the training notebook in `notebooks/` for details.

## Deployment

### Local

```bash
docker-compose up
```

### Cloud Platforms

#### AWS ECS/Fargate
```bash
# Build and push to ECR
docker build -t ner-api .
docker tag ner-api:latest <account>.dkr.ecr.region.amazonaws.com/ner-api:latest
docker push <account>.dkr.ecr.region.amazonaws.com/ner-api:latest
```

#### Google Cloud Run
```bash
gcloud run deploy ner-api --source . --platform managed
```

## Known Limitations

- Model currently detects only Chemicals and Diseases
- Maximum input length: 512 tokens

## License

MIT License - see LICENSE file for details

## Example Results

```
Input: "Aspirin reduces inflammation and treats headaches"

```

![Example 1](ner-project/images/example1.png)
![Example 2](ner-project/images/example2.png)

