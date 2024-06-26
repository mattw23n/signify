import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

# Define the model class (ensure this matches the model used during training)
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
model_path = 'asl_model.pth'
model = ASLModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Define transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Function to preprocess hand landmarks
def preprocess_landmarks(landmarks):
    data = []
    for landmark in landmarks.landmark:
        data.append(landmark.x)
        data.append(landmark.y)
        data.append(landmark.z)
    return np.array(data)

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands
    results = hands.process(image_rgb)
    
    # Draw hand landmarks and classify the hand sign
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Preprocess landmarks
            data = preprocess_landmarks(hand_landmarks)
            data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            
            # Classify the hand sign
            with torch.no_grad():
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                sign = chr(predicted.item() + 65)  # Convert to ASCII (A-Z)
                
                # Display the detected sign on the image
                cv2.putText(image, sign, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the image
    cv2.imshow('Hand Tracking', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
