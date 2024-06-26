import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
from PIL import Image
import mediapipe as mp
import numpy as np
from sklearn.metrics import accuracy_score

# Hardcoded configurations
test_images_dir = 'dataset/test/images'
model_save_path = 'asl_model_retrained.pth'
batch_size = 32

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Define custom dataset class for test dataset
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

# Load the test dataset
test_dataset = ASLDataset(images_dir=test_images_dir)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model class (same as the one used for training)
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

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ASLModel().to(device)
model.load_state_dict(torch.load(model_save_path))
model.eval()

# Evaluate the model on the test dataset
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy
test_accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy: {test_accuracy:.4f}')
