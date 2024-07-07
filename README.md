# Signify
### Team Tik Tak Tok
Matteo, Gilchris, Tania, Justin, Samuel

Project Demo : https://youtu.be/s7-WKfctnF4?si=5KTThWnoYZXNAYCJ

## What Signify Does

Signify empowers deaf and sign language content creators by providing a tool to translate Word-Level American Sign Language (WLASL) into text, generating subtitles for their videos. Additionally, it can incorporate personalized speech into the video, enhancing accessibility and engagement for a broader audience.

## How We Built It

Our application processes videos to recognize and transcribe sign language into text, which is then enhanced and optionally synthesized into the user's voice. The process begins with dissecting the video into individual frames using OpenCV. Each frame is analyzed using MediaPipe to detect 21 human hand landmarks, represented as 3-dimensional vectors for each landmark. These vectors are flattened into a one-dimensional array suitable for machine learning models.

The flattened landmark vectors are fed into a PyTorch model that predicts the word associated with each frame. After several seconds, the generated string of words is sent to a language model (OpenAI in our case) to enhance the text by adding proper punctuation and refining the overall structure.

Optionally, users can synthesize their own voice by recording several voice notes, which help us capture the structure of their voice. Using Tacotron2, a voice synthesizing model, we create a voice file that mimics the user's voice. This synthesized voice, along with subtitles, is then synchronized with the video according to the timestamps from which they were derived. The sign language to text model is built using PyTorch and trained with data we have created for this project.

## Project Setup

### Required Dependencies

#### Frontend
- nativewind
- react native
- react navigation
- axios
- expo
- expo-av
- expo-clipboard
- expo-status-bar
- react-native-modal
- react-native-vector-icons

#### Backend
- Django

### Installation Instructions

1. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   

2. Update IP Addresses:

   Navigate to the following files and replace <YOUR_IP_ADDRESS> with your own IP address:
   - backend/django/navigation/settings.py (line 29)
   - frontend/screens/UploadScreen.js (line 40)
   - frontend/screens/GeneratedScreen.js (line 36)

3. Run the Backend:
   From the base folder, execute the following commands:
   
   cd backend/django
   python manage.py runserver
   

4. Run the Frontend:
   From the base folder, execute the following commands:
   
   cd ../../frontend
   REACT_NATIVE_PACKAGER_HOSTNAME=YOUR_IP_ADDRESS
   npx expo start --lan
   
   Replace YOUR_IP_ADDRESS with your own IP address.

### Notes
- Ensure that all dependencies are properly installed before running the backend and frontend servers.
- Make sure to replace the IP addresses in all mentioned files to avoid connectivity issues.
- For any issues or questions, refer to the official documentation of each dependency.
```
