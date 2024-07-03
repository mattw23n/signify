import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
import json
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Create a label mapping for the alphabet (A-Z)
alphabet_label_mapping = {chr(i): i - 65 for i in range(65, 91)}  # {'A': 0, 'B': 1, ..., 'Z': 25}
reverse_alphabet_label_mapping = {v: k for k, v in alphabet_label_mapping.items()}

# Create a label mapping for WASL
wasl_label_mapping = {'hello': 0, 'thanks': 1, 'thank you': 1, 'yes': 2, 'no': 3}
reverse_wasl_label_mapping = {v: k for k, v in wasl_label_mapping.items()}

# Define the Alphabet Model
class AlphabetModel(nn.Module):
    def __init__(self):
        super(AlphabetModel, self).__init__()
        self.fc1 = nn.Linear(63, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 26)  # 26 classes for A-Z

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the WASL Model
class WASLModel(nn.Module):
    def __init__(self):
        super(WASLModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=63, out_channels=128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 4)  # 4 classes: Hello, Thanks, Yes, No

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, num_features) -> (batch_size, 1, num_features)
        x = torch.relu(self.conv1(x.permute(0, 2, 1)))
        x = x.permute(0, 2, 1)  # permute back for LSTM
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take the last output of the LSTM
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained models
alphabet_model_path = 'models/asl_model.pth'
wasl_model_path = 'models/refined_sign_language_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

alphabet_model = AlphabetModel().to(device)
alphabet_model.load_state_dict(torch.load(alphabet_model_path, map_location=device))
alphabet_model.eval()

wasl_model = WASLModel().to(device)
wasl_model.load_state_dict(torch.load(wasl_model_path, map_location=device))
wasl_model.eval()

# Set the video path
video_path = 'media/upload2.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    print("Video opened successfully.")

# Parameters for movement detection
movement_threshold = 0.15  # Recommended starting point; adjust as needed
previous_landmarks = None

# Parameters for stability check
current_sign = None
current_sign_start_time = None
min_duration_word = 0.275  # Minimum duration (in seconds) to confirm a word
min_duration_letter = 0.8  # Minimum duration (in seconds) to confirm a letter

# Store the results
results_list = []

# Function to calculate the Euclidean distance between two sets of landmarks
def calculate_movement(current_landmarks, previous_landmarks):
    return np.linalg.norm(current_landmarks - previous_landmarks)

# Function to process the video feed and make predictions
def process_video():
    global previous_landmarks, current_sign, current_sign_start_time

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks.append([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            landmarks = np.array(landmarks).flatten()

            # Calculate movement
            if previous_landmarks is not None:
                movement = calculate_movement(landmarks, previous_landmarks)
            else:
                movement = 0

            previous_landmarks = landmarks

            # Prepare landmarks for the models
            landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).to(device).unsqueeze(0)

            # Decide which model to use based on movement
            if movement < movement_threshold:
                # Predict with Alphabet Model
                with torch.no_grad():
                    alphabet_outputs = alphabet_model(landmarks_tensor)
                    alphabet_predicted = torch.argmax(alphabet_outputs, dim=1).item()
                    alphabet_letter = reverse_alphabet_label_mapping[alphabet_predicted]
                    alphabet_confidence = torch.nn.functional.softmax(alphabet_outputs, dim=1)[0][alphabet_predicted].item()

                # Check if the sign is stable
                current_time = time.time()
                if current_sign == alphabet_letter and alphabet_confidence > 0.2:
                    if current_time - current_sign_start_time >= min_duration_letter:
                        results_list.append(alphabet_letter)
                        current_sign = None  # Reset to detect new signs
                        print(f"Recorded letter: {alphabet_letter}")
                else:
                    current_sign = alphabet_letter
                    current_sign_start_time = current_time
                    print(f"Detected new letter: {alphabet_letter}")

                # Display the predictions on the frame
                cv2.putText(frame, f'Letter: {alphabet_letter} ({alphabet_confidence:.2f})', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # Predict with WASL Model
                with torch.no_grad():
                    wasl_outputs = wasl_model(landmarks_tensor)
                    wasl_predicted = torch.argmax(wasl_outputs, dim=1).item()
                    wasl_sign = reverse_wasl_label_mapping[wasl_predicted]
                    wasl_confidence = torch.nn.functional.softmax(wasl_outputs, dim=1)[0][wasl_predicted].item()

                # Check if the sign is stable
                current_time = time.time()
                if current_sign == wasl_sign:
                    if current_time - current_sign_start_time >= min_duration_word:
                        results_list.append(wasl_sign)
                        current_sign = None  # Reset to detect new signs
                        print(f"Recorded sign: {wasl_sign}")
                else:
                    current_sign = wasl_sign
                    current_sign_start_time = current_time
                    print(f"Detected new sign: {wasl_sign}")

                # Display the predictions on the frame
                cv2.putText(frame, f'Sign: {wasl_sign} ({wasl_confidence:.2f})', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('ASL and WASL Translation', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Check if results_list is populated and save to JSON
    if results_list:
        with open('results.json', 'w') as f:
            json.dump(results_list, f, indent=4)
        print("Results saved to results.json")
    else:
        print("No signs recorded.")

    print('Process completed.')

# Start the video processing
process_video()
