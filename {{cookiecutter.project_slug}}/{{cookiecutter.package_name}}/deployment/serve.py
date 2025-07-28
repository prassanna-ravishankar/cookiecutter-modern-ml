import logging
from pathlib import Path
from typing import Dict, List

import litserve as ls
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from {{ cookiecutter.package_name }}.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextClassificationAPI(ls.LitAPI):
    def setup(self, device: str):
        settings = get_settings()
        
        # Find the latest model directory
        model_dir = Path("models") / settings.model.checkpoint.replace("/", "-")
        if not model_dir.exists():
            raise ValueError(
                f"Model directory {model_dir} not found. Please train the model first."
            )
        
        logger.info(f"Loading model from {model_dir}")
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        
        # Move model to device
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
        # Load settings
        self.max_length = settings.model.max_length
        
        # Label mappings for IMDB
        self.id2label = {0: "negative", 1: "positive"}
    
    def decode_request(self, request: Dict) -> str:
        return request["text"]
    
    def predict(self, text: str) -> Dict:
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
        
        # Convert to Python floats
        probs = probabilities[0].cpu().numpy().tolist()
        
        return {
            "prediction": self.id2label[predicted_class],
            "confidence": float(probs[predicted_class]),
            "probabilities": {
                self.id2label[i]: float(probs[i]) for i in range(len(probs))
            },
        }
    
    def encode_response(self, output: Dict) -> Dict:
        return output


def main():
    settings = get_settings()
    
    # Create API instance
    api = TextClassificationAPI()
    
    # Create server
    server = ls.LitServer(
        api,
        accelerator="auto",
        devices=1,
        workers_per_device=settings.serving.workers,
    )
    
    # Run server
    logger.info(f"Starting server on {settings.serving.host}:{settings.serving.port}")
    server.run(host=settings.serving.host, port=settings.serving.port)


if __name__ == "__main__":
    main()