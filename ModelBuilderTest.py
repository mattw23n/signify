import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import mediapipe as mp
import numpy as np
from sklearn.metrics import accuracy_score

# Hardcoded configurations
train_images_dir = 'dataset/train/images'
val_images_dir = 'dataset/valid/images'
model_save_path = 'asl_model.pth'
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Define custom dataset class
class ASLDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

        print(f"Number of images in {images_dir}: {len(self.image_files)}")  # Debugging

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Extract label from filename (assuming label is the first character of the filename)
        label = img_name[0].upper()
        label = ord(label) - 65  # Convert label from 'A' to 0, 'B' to 1, ..., 'Z' to 25

        # Convert PIL image to OpenCV format
        image_cv = np.array(image)

        # Process the image with MediaPipe
        results = hands.process(image_cv)

        # If hand landmarks are detected, use them
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).flatten()
        else:
            # If no landmarks detected, use a zero array (or handle accordingly)
            landmarks = np.zeros(21 * 3)

        # Convert landmarks to tensor
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)

        return landmarks_tensor, label

# Load datasets
train_dataset = ASLDataset(images_dir=train_images_dir)
val_dataset = ASLDataset(images_dir=val_images_dir)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the model
class ASLModel(nn.Module):
    def __init__(self):
        super(ASLModel, self).__init__()
        self.fc1 = nn.Linear(21 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 26)  # 26 classes for A-Z

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Move the model to device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ASLModel().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Validation phase
    model.eval()
    val_loss = 0.0
    corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            corrects += torch.sum(preds == labels)
    val_loss /= len(val_loader.dataset)
    val_acc = corrects.double() / len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')

# Save the trained model
torch.save(model.state_dict(), model_save_path)
