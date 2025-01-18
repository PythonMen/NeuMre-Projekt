import os
import kagglehub
from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Download dataset
print("Downloading dataset...")
dataset_path = kagglehub.dataset_download("kausthubkannan/5-flower-types-classification-dataset")
print("Dataset downloaded to:", dataset_path)

# Define paths for training and testing directories
data_dir = os.path.join(dataset_path, "flower_images")

# Data transformations
print("Transforming dataset")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split dataset into training and testing
print("Splitting Dataset")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Dataloaders
print("Establishing dataloaders")
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Run experiments
model = models.resnet34(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))  # Match output to number of classes
model = model.to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

# Evaluation Loop
def evaluate(model, loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100 * correct / total

# Train and Evaluate
for epoch in range(15):
    train_loss = train(model, train_loader, criterion, optimizer)
    test_accuracy = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

