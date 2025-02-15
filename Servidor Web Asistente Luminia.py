from flask import Flask, render_template, request, jsonify, Response
from huggingface_hub import InferenceClient
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
#from dimits import Dimits
import os, base64, random, pyttsx3

CHAT_MODEL = "Qwen/Qwen2.5-72B-Instruct"
AUDIO_RECOGNITION_MODEL = "openai/whisper-large-v3-turbo"
RAG_MODEL_ONLINE = "intfloat/multilingual-e5-large-instruct"
HF_TOKEN = "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"
DIRECTORIO = os.path.dirname(os.path.realpath(__file__))
CHROMA_PATH = DIRECTORIO + "/Base de Datos Vectorial Chroma"
TTS_MODEL_PATH = "/voices/es_MX-claude-high"
PROMPT_TEMPLATE = """
Responde la anterior pregunta basado solamente en el siguiente contexto:

{context}

---

Responde la anterior pregunta basada en el contexto de arriba: {question}
"""

#engine = Dimits(DIRECTORIO + TTS_MODEL_PATH)
engine = pyttsx3.init()
engine.setProperty("rate", 160)

app = Flask(__name__)

client = InferenceClient(model=CHAT_MODEL, token=HF_TOKEN)
stt_client = InferenceClient(model=AUDIO_RECOGNITION_MODEL, token=HF_TOKEN)

embedding_function = HuggingFaceInferenceAPIEmbeddings(model_name=RAG_MODEL_ONLINE, api_key=HF_TOKEN)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
conversation = [{"role": "system", "content":
"""Your name is Luminia and you speak only in Spanish, you're an assistant for helping students in the university Tecnologic Institute of Costa
Grande with the information provided to you, giving short and preccise answers.

When someone asks for information about something or the location of some departament or places, if possible assume it's information from the
TecNM aka the Tecnologic Institute of Costa Grande, and respond only with: [RAG_SEARCH], nothing else, otherwise respond normally if no
question is asked.

After the proper context is given to you, try to answer the request from the student, if none of the information provided to you can
answer the question simply say that the information couldn't be found, maybe it's not in the database and probably should ask a person from the
institute instead.

In your responses avoid at all cost placing emojis and any unreadable stuff without proper context, your responses are read aloud by a text to speech
application, so avoid placing any unreadable stuff in your response, additionally, mathematical expressions such as x^2 put them as x squared
and such so the text to speech can pronounce them properly."""},
    {'role': 'assistant', 'content': 'Hola estudiante, ¿En que puedo servirte el dia de hoy?'}]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query_audio', methods=['POST'])
def query_audio():
    audio = request.files['audio']

    return jsonify({"response": stt_client.automatic_speech_recognition(audio).text})

@app.route('/send_audio', methods=['POST'])
def send_audio():
    audio = request.files['audio']

    return jsonify({"response": stt_client.automatic_speech_recognition(audio).text})

@app.route('/chat_stream', methods=['POST'])
def chat_stream():
    user_message = request.json.get("message")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    
    conversation.append({"role": "user", "content": user_message})

    return jsonify({"response": "OK"})

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    data = request.json
    message = data.get("text")
    count = data.get("count")

    #engine.text_2_audio_file(message, f"audios/speech{count}", DIRECTORIO, format="wav")
    engine.save_to_file(message, f"{DIRECTORIO}/audios/speech{count}.wav")
    engine.runAndWait()

    with open(DIRECTORIO + f"/audios/speech{count}.wav", "rb") as audio_file:
        audio_data = base64.b64encode(audio_file.read()).decode('utf-8')

    return jsonify({"audio": audio_data, "position": count})

@app.route('/listen_stream')
def listen_stream():
    def AI_response():
        completion = client.chat_completion(conversation, stream=True, max_tokens=500, seed=random.randint(1000000, 10000000))
        response = ""

        try:
            for chunk in completion:
                if (chunk.choices[0].delta.content is not None):
                    part = chunk.choices[0].delta.content
                    response += part

                    yield f"data: {part}.\n\n"

                    if ("[RAG_SEARCH]" in response):
                        break
        except:
            pass
        
        if ("[RAG_SEARCH]" in response):
            user_prompt = conversation[len(conversation) - 1]["content"]
            results = db.similarity_search_with_score(user_prompt, k=10)
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt = PROMPT_TEMPLATE.format(context=context_text, question=user_prompt)

            yield "data: [RAG_FINISHED].\n\n"

            completion = client.chat_completion([{"role": "user", "content": prompt}], stream=True, max_tokens=500)
            response = ""

            try:
                for chunk in completion:
                    if (chunk.choices[0].delta.content is not None):
                        part = chunk.choices[0].delta.content
                        response += part

                        yield f"data: {part}.\n\n"
            except:
                pass

        conversation.append({"role": "assistant", "content": response})

    return Response(AI_response(), content_type='text/event-stream')
        

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
