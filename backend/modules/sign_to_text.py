import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
import json
import time
import logging

logger = logging.getLogger(__name__)

project_dir = os.path.dirname(os.path.dirname(__file__))

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)

wasl_label_mapping = {
    'hello': 0,
    'going-to': 1,
    'create': 2,
    'content': 3,
    'i-am': 4,
    'learn': 5,
    'sign-language': 6,
    'today': 7,
    'using': 8,
    'how-to': 9
}
reverse_wasl_label_mapping = {v: k for k, v in wasl_label_mapping.items()}

class WASLModel(nn.Module):
    def __init__(self):
        super(WASLModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=225, out_channels=128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, len(wasl_label_mapping))

    def forward(self, x):
        x = torch.relu(self.conv1(x.permute(0, 2, 1)))
        x = x.permute(0, 2, 1)  
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model_save_path = os.path.join(project_dir, 'models', 'WASL_sign_language_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WASLModel().to(device)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

video_path = os.path.join(project_dir, 'video', 'data_2.mp4')
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    logger.error("Error: Could not open video.")
else:
    logger.info("Video opened successfully.")

results_list = []

start_time = time.time()

def process_video():
    global start_time
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        landmarks = []
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0, 0.0, 0.0] * 21)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0, 0.0, 0.0] * 21)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark[:33]:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0, 0.0, 0.0] * 33)

        if len(landmarks) == 225:
            landmarks = np.array(landmarks).reshape(1, -1, 225)

            landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).to(device)

            with torch.no_grad():
                outputs = model(landmarks_tensor)
                confidence, predicted = torch.max(outputs, 1)
                sign = reverse_wasl_label_mapping[predicted.item()]
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()

                cv2.putText(frame, f'{sign} ({confidence:.2f})', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                current_time = time.time()
                if current_time - start_time >= 0.5:
                    results_list.append({'sign': sign, 'confidence': confidence})
                    start_time = current_time

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow('WASL Translation', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    results_file_path = os.path.join(project_dir, 'video', 'results.json')
    if results_list:
        with open(results_file_path, 'w') as f:
            json.dump(results_list, f, indent=4)
        logger.info("Results saved to results.json")
    else:
        logger.warning("No signs recorded.")


