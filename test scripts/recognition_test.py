from huggingface_hub import InferenceClient
import speech_recognition as sr, os

client = InferenceClient(
    model="openai/whisper-large-v3-turbo",
    token="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",
)

recognizer = sr.Recognizer()
with sr.Microphone() as source:
    print("Listening")
    audio = recognizer.listen(source, timeout=3, phrase_time_limit=10)
    print("Done")

text = client.automatic_speech_recognition(audio.get_wav_data()).text

print(text)