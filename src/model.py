import torch.nn as nn
from torchvision import models


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """
    Create and return the image classification model.

    Args:
        architecture: Model architecture to use.
        num_classes: Number of output classes.

    Returns:
        PyTorch model.
    """

    if architecture == "resnet18":
        model = models.resnet18(weights=None)

        # CIFAR-10 images are 32x32.
        # Use a smaller initial convolution and remove the
        # initial max-pooling layer used for larger images.
        model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        model.maxpool = nn.Identity()

        # CIFAR-10 has 10 classes.
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

        return model

    raise ValueError(f"Unsupported architecture: {architecture}")