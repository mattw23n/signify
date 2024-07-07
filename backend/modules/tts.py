import os
from gtts import gTTS

def text_to_speech(sentence):

    save_path = "./resources/output.mp3"
    directory = os.path.dirname(save_path)
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    tts = gTTS(sentence, lang='en')
    tts.save(save_path)
    print(f"Speech has been saved to {save_path}")


