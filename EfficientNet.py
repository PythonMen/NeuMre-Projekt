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
results = []
for num_epochs in iterations:
    for version in versions:
        print(f"Training EfficientNet-{version} incrementally for {num_epochs} epochs...")

        # Load model
        model = get_efficientnet_model(version, num_classes=len(dataset.classes))
        model = model.to(device)

        # Get optimizer and loss function
        optimizer = get_optimizer(model, lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate incrementally
        start_epoch = 0
        for i, epoch_count in enumerate(iterations):
            print(f"Training for {epoch_count} more epochs (epoch {start_epoch + 1} to {start_epoch + epoch_count})")
            train_model_incremental(model, train_loader, criterion, optimizer, device, start_epoch, epoch_count)

            # Test model after each stage
            print(f"Evaluating EfficientNet-{version} after {start_epoch + epoch_count} epochs...")
            accuracy = evaluate_model(model, test_loader, device)

            # Store results
            results.append((version, start_epoch + epoch_count, accuracy))

            # Increment the start_epoch for the next phase
            start_epoch += epoch_count

# Print all results
print("\nFinal Results:")
for version, num_epochs, accuracy in results:
    print(f"EfficientNet-{version}, Epochs: {num_epochs}, Test Accuracy: {accuracy:.2f}%")
