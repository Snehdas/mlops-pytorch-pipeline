import io
import os

import torch
from PIL import Image
from fastapi.testclient import TestClient

from src.model import get_model
from src import serve


# ---------------------------------------------------------
# Test 1 — Model creation
# ---------------------------------------------------------
def test_model_creation():
    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    assert model is not None
    assert isinstance(model, torch.nn.Module)


# ---------------------------------------------------------
# Test 2 — Model output
# ---------------------------------------------------------
def test_model_output():
    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    model.eval()

    # CIFAR-10 image:
    # batch size = 1
    # channels = 3
    # height = 32
    # width = 32
    x = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 10)


# ---------------------------------------------------------
# Test 3 — Checkpoint loading
# ---------------------------------------------------------
def test_checkpoint_loading():
    checkpoint_path = "checkpoints/classifier_v1.pt"

    # The checkpoint is created during Day 3 training.
    # Skip this test if no checkpoint exists locally.
    if not os.path.exists(checkpoint_path):
        return

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    assert checkpoint is not None
    assert "model_state_dict" in checkpoint

    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    assert model is not None


# ---------------------------------------------------------
# Test 4 — API health
# ---------------------------------------------------------
def test_api_health(monkeypatch):
    # Use a model directly so the test does not depend
    # on the checkpoint being present in CI.
    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    model.eval()

    monkeypatch.setattr(
        serve,
        "model",
        model,
    )

    client = TestClient(serve.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


# ---------------------------------------------------------
# Test 5 — API prediction
# ---------------------------------------------------------
def test_api_prediction(monkeypatch):
    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    model.eval()

    monkeypatch.setattr(
        serve,
        "model",
        model,
    )

    client = TestClient(serve.app)

    # Create a dummy 32x32 RGB image.
    image = Image.new(
        "RGB",
        (32, 32),
        color=(128, 128, 128),
    )

    image_bytes = io.BytesIO()

    image.save(
        image_bytes,
        format="PNG",
    )

    image_bytes.seek(0)

    response = client.post(
        "/predict",
        files={
            "image": (
                "test.png",
                image_bytes,
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert "class" in result
    assert "class_index" in result
    assert "probabilities" in result

    assert 0 <= result["class_index"] < 10

    assert len(
        result["probabilities"]
    ) == 10