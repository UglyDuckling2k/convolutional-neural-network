import os
import time
import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
model_path = "./model/model.pth"
data_path = "./data"

# Configure logging
logging.basicConfig(level=logging.INFO)

# Suppress specific warning
warnings.filterwarnings("ignore", message="Truncated File Read")

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Set Seed
np.random.seed(42)
torch.manual_seed(42)

# Image transformations
train_transform = transforms.Compose(
    [
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

inv_normalize_transform = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


# Load data
train_data = datasets.ImageFolder(
    os.path.join(data_path, "train"), transform=train_transform
)

test_data = datasets.ImageFolder(
    os.path.join(data_path, "test"), transform=test_transform
)

train_loader = DataLoader(train_data, batch_size=100, num_workers=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100,  num_workers=8, shuffle=False)

class_names = train_data.classes

# Display images for debugging
# for images,labels in train_loader:
#     break
# im = make_grid(images, nrow=5)
# im_inv = inv_normalize_transform(im)
# plt.figure(figsize=(12,4))
# plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)))
# plt.show()


# Create model
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)

        # Assuming the input image size is 224x224, adjust the following line if the input size is different
        self.fc1 = nn.Linear(54 * 54 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54 * 54 * 16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)


# Setup model
model = ConvolutionalNetwork().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Train model
epochs = 6

train_losses = []
train_correct = []
test_losses = []
test_correct = []

start_time = time.time()

for epoch in range(epochs):
    epoch_train_correct = 0
    epoch_test_correct = 0
    epoch_loss = 0 
    
    # Set model to train mode
    model.train()
    
    for batch_index, (images, labels) in enumerate(train_loader):

        # Move data & labels to CUDA device
        images, labels = images.to(device), labels.to(device)

        # Forward pass: compute the model output
        output = model(images)
        loss = criterion(output, labels)

        # Calculate accuracy
        predictions = torch.max(output.data, 1)[1]
        batch_train_correct = (predictions == labels).sum()
        epoch_train_correct += batch_train_correct

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log progress every 200 batches
        if batch_index % 100 == 0:
            logging.info(
                f"Training | Epoch: {epoch:2} | Batch: {batch_index:4} | Loss: {loss.item():.8f} | Correct: {batch_train_correct}/{len(images)}"
            )

    # Record training loss and accuracy for the epoch
    train_losses.append(loss.item())
    train_correct.append(epoch_train_correct)

    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_loader):

            # Move data & labels to CUDA device
            images, labels = images.to(device), labels.to(device)

            # Forward pass to get output
            output = model(images)
            loss = criterion(output, labels)
            
            # Accumulate loss over all batches
            epoch_loss += loss.item()

            # Get predictions from the maximum value
            predictions = torch.max(output, 1)[1]

             # Calculate accuracy
            batch_test_correct = (predictions == labels).sum()
            epoch_test_correct += batch_test_correct

        # Calculate average loss for the epoch
        avg_test_loss = epoch_loss / len(test_loader)
   
        # Record test loss and accuracy
        test_losses.append(avg_test_loss)
        test_correct.append(epoch_test_correct)
        
        logging.info(
            f"Testing  | Epoch: {epoch:2} |             | Loss: {avg_test_loss:.8f} | Correct: {epoch_test_correct}/{len(test_loader.dataset)}"
        )


# Log total training duration
total_duration_time = time.time() - start_time
duration_minutes, duration_seconds = divmod(total_duration_time, 60)
logging.info(
    f"Training took {int(duration_minutes)} minutes and {int(duration_seconds)} seconds"
)

# Save model to disk
torch.save(model.state_dict(), model_path)
logging.info("Model saved successfully.")

# Convert tensor to numpy array and ensure it's on CPU for plotting
train_accuracy = [(100 * t.item() / len(train_loader.dataset)) for t in train_correct]
test_accuracy = [(100 * t.item() / len(test_loader.dataset)) for t in test_correct]

logging.info(
    f"Final Accuracy: {100 * test_correct[-1].item() / len(test_loader.dataset)}"
)

# Plot training and test losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training loss")
plt.plot(test_losses, label="Test loss")
plt.title("Loss at the end of each epoch")
plt.legend()
plt.show(block=False)

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracy, label="Training accuracy")
plt.plot(test_accuracy, label="Test accuracy")
plt.title("Accuracy at the end of each epoch")
plt.legend()
plt.show()
