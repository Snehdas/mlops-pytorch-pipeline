import os
import io

import torch
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image

from src.model import get_model


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "./checkpoints/classifier_v1.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

app = FastAPI(
    title="CIFAR-10 Image Classification API",
    version="1.0.0"
)


# CIFAR-10 class names
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# Load model
model = get_model()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "resnet18",
        "device": str(DEVICE)
    }


@app.post("/predict")
def predict(image: UploadFile = File(...)):

    try:
        # Read uploaded image
        image_bytes = image.file.read()

        # Convert uploaded image to RGB
        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # CIFAR-10 preprocessing
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])

        image_tensor = transform(pil_image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(DEVICE)

        # Prediction
        with torch.no_grad():
            outputs = model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

        class_id = predicted.item()

        predicted_class = CLASS_NAMES[class_id]

        return {
            "prediction": predicted_class,
            "class_id": class_id,
            "confidence": round(
                confidence.item(),
                4
            )
        }

    except Exception as e:
        return {
            "error": str(e)
        }

