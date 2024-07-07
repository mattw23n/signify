from gtts import gTTS

def text_to_speech(sentence):
    tts = gTTS(sentence, lang='en')
    tts.save("backend/resources/output.mp3")
    print("Speech has been saved to backend/resources/output.mp3")

