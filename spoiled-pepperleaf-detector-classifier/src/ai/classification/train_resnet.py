#!/usr/bin/env python3
"""
Unified Korean Pepper Leaf Spoilage Classifier trainer.

Example:
    python train_resnet.py --backbone resnet50
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ARCH_CONFIG = {
    "resnet18": {
        "builder": models.resnet18,
        "hidden_features": 256,
        "num_epochs": 150,
        "patience": 15,
        "model_filename": "leaf_classifier_resnet18.pth",
        "plot_prefix": "leaf_resnet18",
    },
    "resnet50": {
        "builder": models.resnet50,
        "hidden_features": 512,
        "num_epochs": 100,
        "patience": 10,
        "model_filename": "leaf_classifier_resnet50.pth",
        "plot_prefix": "leaf_resnet50",
    },
}


class LeafDataset(Dataset):
    """Custom dataset for leaf images."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_image_paths_and_labels(data_dir: Path):
    """Return flattened list of image paths with binary labels."""
    image_paths = []
    labels = []

    normal_dir = data_dir / "leaf" / "normal" / "normal"
    if normal_dir.exists():
        for img_file in normal_dir.iterdir():
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_paths.append(str(img_file))
                labels.append(0)

    spoiled_dir = data_dir / "leaf" / "spoiled"
    if spoiled_dir.exists():
        for harm_path in spoiled_dir.iterdir():
            if harm_path.is_dir():
                for img_file in harm_path.iterdir():
                    if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        image_paths.append(str(img_file))
                        labels.append(1)

    return image_paths, labels


def create_data_loaders(
    data_dir: Path,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
):
    """Create train, validation, and test data loaders."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    image_paths, labels = get_image_paths_and_labels(data_dir)

    print(f"Total images found: {len(image_paths)}")
    print(f"Normal images: {labels.count(0)}")
    print(f"Spoiled images: {labels.count(1)}")

    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths, labels = zip(*combined)

    train_idx = int(len(image_paths) * train_split)
    val_idx = train_idx + int(len(image_paths) * val_split)

    train_paths = image_paths[:train_idx]
    train_labels = labels[:train_idx]
    val_paths = image_paths[train_idx:val_idx]
    val_labels = labels[train_idx:val_idx]
    test_paths = image_paths[val_idx:]
    test_labels = labels[val_idx:]

    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    train_dataset = LeafDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = LeafDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = LeafDataset(test_paths, test_labels, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader


def create_model(backbone: str, num_classes: int = 2, pretrained: bool = True):
    """Create the requested ResNet backbone with a custom classifier head."""
    if backbone not in ARCH_CONFIG:
        raise ValueError(f"Unsupported backbone '{backbone}'")

    config = ARCH_CONFIG[backbone]
    model = config["builder"](pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features
    hidden = config["hidden_features"]
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, num_classes),
    )
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, all_predictions, all_targets


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Training Loss")
    ax1.plot(epochs, val_losses, label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xticks(epochs[:: max(1, len(epochs) // 10)])

    ax2.plot(epochs, train_accs, label="Training Accuracy")
    ax2.plot(epochs, val_accs, label="Validation Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)
    ax2.set_xticks(epochs[:: max(1, len(epochs) // 10)])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Spoiled"],
        yticklabels=["Normal", "Spoiled"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(save_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train pepper leaf classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/data/processed/classification"),
        help="Root directory of the prepared dataset.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/weights/trained"),
        help="Directory where checkpoints will be saved.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/results"),
        help="Directory where training plots will be written.",
    )
    parser.add_argument(
        "--backbone",
        choices=ARCH_CONFIG.keys(),
        default="resnet18",
        help="Which ResNet backbone to fine-tune.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument(
        "--train-split", type=float, default=0.7, help="Fraction of data for training."
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15, help="Fraction of data for validation."
    )
    parser.add_argument(
        "--test-split", type=float, default=0.15, help="Fraction of data for testing."
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Seed for reproducibility."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = ARCH_CONFIG[args.backbone]
    args.model_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / args.backbone).mkdir(parents=True, exist_ok=True)

    model_save_path = args.model_dir / config["model_filename"]
    results_dir = args.results_dir / args.backbone

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Selected backbone: {args.backbone}")

    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        random_seed=args.random_seed,
    )

    print(f"Creating {args.backbone} model...")
    model = create_model(backbone=args.backbone, pretrained=True, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = config["num_epochs"]
    patience = config["patience"]
    best_val_acc = 0.0
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("Starting training...")
    print("=" * 50)

    training_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - training_start_time

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(
            f"Epoch Time: {epoch_time:.2f}s, Total Time: {str(timedelta(seconds=int(total_time)))}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "backbone": args.backbone,
                },
                model_save_path,
            )
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")

        if patience_counter >= patience:
            total_training_time = time.time() - training_start_time
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            print(
                f"Total training time: {str(timedelta(seconds=int(total_training_time)))}"
            )
            break

        print()

    print("Final evaluation on test set...")
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_preds, test_targets = validate_epoch(
        model, test_loader, criterion, device
    )

    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    print("\nTest Set Classification Report:")
    print(classification_report(test_targets, test_preds, target_names=["Normal", "Spoiled"]))

    print("\nValidation Set Results (for comparison):")
    val_loss, val_acc, val_preds, val_targets = validate_epoch(
        model, val_loader, criterion, device
    )
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")

    plot_prefix = config["plot_prefix"]
    plot_training_history(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        results_dir / f"{plot_prefix}_training_history.png",
    )
    plot_confusion_matrix(
        val_targets, val_preds, results_dir / f"{plot_prefix}_confusion_matrix_validation.png"
    )
    plot_confusion_matrix(
        test_targets, test_preds, results_dir / f"{plot_prefix}_confusion_matrix_test.png"
    )

    total_training_time = time.time() - training_start_time

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Total training time: {str(timedelta(seconds=int(total_training_time)))}")
    print(f"Model saved to: {model_save_path}")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
