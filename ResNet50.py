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

# Function to modify and load EfficientNet model
def get_efficientnet_model(version, num_classes):
    if version == "b0":
        model = models.efficientnet_b0(pretrained=True)
    elif version == "b1":
        model = models.efficientnet_b1(pretrained=True)
    elif version == "b2":
        model = models.efficientnet_b2(pretrained=True)
    else:
        raise ValueError("Unsupported EfficientNet version")

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# Define loss function and optimizer
def get_optimizer(model, lr=0.001):
    return optim.Adam(model.parameters(), lr=lr)

# Training loop with incremental learning
def train_model_incremental(model, train_loader, criterion, optimizer, device, start_epoch, num_epochs):
    model.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{start_epoch + num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Experiment configurations
iterations = [5, 5, 5]  # Incremental training: 5 + 5 + 5 epochs
versions = ["b0", "b1", "b2"]  # EfficientNet versions

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







# Run experiments

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, model.fc.out_features)  # Match output to number of classes
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

