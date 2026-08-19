import os
import sys
from io import BytesIO

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

# Allow imports when running:
# uvicorn src.serve:app
sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

from model import get_model


app = FastAPI(
    title="MLOps PyTorch Classifier",
    version="1.0.0",
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


# Use GPU if available, otherwise CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Checkpoint location
CHECKPOINT_PATH = os.environ.get(
    "MODEL_CHECKPOINT",
    "checkpoints/classifier_v1.pt",
)


model = None


def load_model():
    global model

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()


# Load checkpoint when application starts
try:
    load_model()
except FileNotFoundError:
    model = None


# CIFAR-10 preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ]
)


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return {
        "status": "ok"
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...)
):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    try:
        image_bytes = await image.read()

        pil_image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        )

    input_tensor = transform(
        pil_image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )[0]

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    return {
        "class": CLASS_NAMES[predicted_index],
        "class_index": predicted_index,
        "probabilities": {
            CLASS_NAMES[i]: round(
                float(probabilities[i]),
                6,
            )
            for i in range(len(CLASS_NAMES))
        },
    }