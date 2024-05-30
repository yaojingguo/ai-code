import azure.cognitiveservices.speech as speechsdk
import os

def from_file():
    subscription=os.environ.get('SPEECH_KEY')
    region=os.environ.get('SPEECH_REGION')
    speech_config = speechsdk.SpeechConfig(subscription="fdc7fd48d7e84abf88ea9bb040457293", region="chinanorth2")
    audio_config = speechsdk.AudioConfig(filename="/Users/jing/Downloads/1.wav")
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    result = speech_recognizer.recognize_once_async().get()
    print(result.text)

from_file()