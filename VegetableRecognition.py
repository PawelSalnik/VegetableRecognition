
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import gradio as gr
from PIL import Image
import os
import numpy as np

# ------------------ Config ------------------
DATA_DIR = "./veggies"  # Folder with subfolders like broccoli/, carrot/, etc.
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_PATH = "veggie_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(os.listdir(DATA_DIR + "/train"))  # assumes train/ is organized into class folders

# ------------------ Transforms ------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------ Datasets & Loaders ------------------
train_dataset = datasets.ImageFolder(DATA_DIR + "/train", transform=transform_train)
test_dataset = datasets.ImageFolder(DATA_DIR + "/test", transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

class_names = train_dataset.classes

# ------------------ Model ------------------
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ------------------ Loss & Optimizer ------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# ------------------ Training ------------------
if not os.path.exists(MODEL_PATH):
    print("Training model...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved!")
else:
    print("Loading model from file...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# ------------------ Evaluation ------------------
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

evaluate()

# ------------------ Gradio Inference ------------------
def predict(img):
    model.eval()
    image = transform_test(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, pred_idx = torch.max(probs, 0)
        label = class_names[pred_idx]

        # Threshold for unknown detection
        if conf.item() < 0.6:
            return "Unknown vegetable (confidence too low)"

    return f"{label.capitalize()} ({conf.item() * 100:.1f}% confidence)"

# ------------------ Gradio App ------------------
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Vegetable Recognition",
    description="Upload a photo of a vegetable. The model will try to recognize it.",
    live=False
)

interface.launch()
