import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.quantization import resnet18


ROOT_DIR = Path(__file__).resolve().parent
SAVE_DIR = ROOT_DIR / "saved_models"
RESULTS_DIR = ROOT_DIR / "results"
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    roots = [
        ROOT_DIR.parent / "data" / "cifar-10-batches-py",
        ROOT_DIR / "data" / "cifar-10-batches-py",
    ]
    data_root = next((path for path in roots if path.is_dir()), None)
    if data_root is None:
        raise FileNotFoundError(
            "CIFAR-10 was not found. See README.md for the expected data folder."
        )

    def read(files):
        batches = []
        for filename in files:
            with (data_root / filename).open("rb") as stream:
                batches.append(pickle.load(stream, encoding="bytes"))
        images = torch.tensor(
            np.concatenate([batch[b"data"] for batch in batches])
        ).view(-1, 3, 32, 32) / 255
        labels = torch.tensor(
            sum([batch[b"labels"] for batch in batches], [])
        )
        return (images - MEAN) / STD, labels

    x_train, y_train = read(
        [f"data_batch_{index}" for index in range(1, 6)]
    )
    x_test, y_test = read(["test_batch"])
    with (data_root / "batches.meta").open("rb") as stream:
        labels = [
            label.decode()
            for label in pickle.load(stream, encoding="bytes")[b"label_names"]
        ]
    return x_train, y_train, x_test, y_test, labels


def build_model():
    model = resnet18(weights=None, quantize=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def load_baseline():
    checkpoint = SAVE_DIR / "fp32_weights.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Baseline checkpoint is missing. Run the baseline-training cell "
            "in Assignment3.ipynb first."
        )
    model = build_model()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model


def test_model(model, x_test, y_test, device="cpu"):
    model.eval().to(device)
    predictions = []
    with torch.no_grad():
        for images in x_test.split(1000):
            predictions.append(model(images.to(device)).argmax(1).cpu())
    predictions = torch.cat(predictions)
    accuracy = round((predictions == y_test).float().mean().item() * 100, 2)
    return accuracy, predictions


def fit_model(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs,
    learning_rate,
    before_epoch=None,
):
    x_gpu = x_train.to(DEVICE)
    y_gpu = y_train.to(DEVICE)
    model.to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4,
    )
    steps = len(x_gpu) // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, learning_rate, total_steps=epochs * steps
    )
    for epoch in range(epochs):
        if before_epoch:
            before_epoch(model, epoch)
        model.train()
        order = torch.randperm(len(x_gpu), device=DEVICE)
        for batch in range(steps):
            indices = order[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
            images = torch.where(
                torch.rand(len(indices), 1, 1, 1, device=DEVICE) < 0.5,
                x_gpu[indices].flip(3),
                x_gpu[indices],
            )
            images = torch.roll(
                images,
                (random.randint(-4, 4), random.randint(-4, 4)),
                (2, 3),
            )
            loss = F.cross_entropy(
                model(images), y_gpu[indices], label_smoothing=0.1
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        accuracy, _ = test_model(model, x_test, y_test, DEVICE)
        print(
            f"  epoch {epoch + 1}/{epochs}: loss {loss.item():.3f}, "
            f"test acc {accuracy}%"
        )
