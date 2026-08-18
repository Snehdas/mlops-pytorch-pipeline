import json
import os

import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import get_model


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device,
) -> tuple[float, float]:
    """Train the model for one epoch."""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate(
    model: nn.Module,
    val_loader,
    criterion,
    device,
) -> tuple[float, float]:
    """Evaluate the model on the validation dataset."""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / total
    val_accuracy = correct / total

    return val_loss, val_accuracy


def main():
    config_path = os.environ.get(
        "TRAINING_CONFIG",
        "configs/training_config.yaml",
    )

    config = load_config(config_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        json.dumps(
            {
                "event": "training_started",
                "device": str(device),
            }
        )
    )

    # Configuration
    architecture = config["model"]["architecture"]
    num_classes = config["model"]["num_classes"]

    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    early_stopping_patience = config["training"]["early_stopping_patience"]

    data_dir = config["data"]["data_dir"]
    num_workers = config["data"].get("num_workers", 2)

    checkpoint_dir = config["output"]["checkpoint_dir"]
    model_name = config["output"]["model_name"]

    # Create output directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Data
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Model
    model = get_model(
        architecture=architecture,
        num_classes=num_classes,
    )

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    checkpoint_path = os.path.join(
        checkpoint_dir,
        model_name,
    )

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # JSON-line metrics
        metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_accuracy, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_accuracy, 6),
        }

        print(json.dumps(metrics))

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": architecture,
                    "num_classes": num_classes,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )

            print(
                json.dumps(
                    {
                        "event": "checkpoint_saved",
                        "path": checkpoint_path,
                        "epoch": epoch,
                    }
                )
            )

        else:
            patience_counter += 1

            print(
                json.dumps(
                    {
                        "event": "early_stopping_counter",
                        "patience": patience_counter,
                        "max_patience": early_stopping_patience,
                    }
                )
            )

            if patience_counter >= early_stopping_patience:
                print(
                    json.dumps(
                        {
                            "event": "early_stopping",
                            "epoch": epoch,
                        }
                    )
                )
                break

    print(
        json.dumps(
            {
                "event": "training_completed",
                "checkpoint": checkpoint_path,
                "best_val_loss": best_val_loss,
            }
        )
    )


if __name__ == "__main__":
    main()