import os, math, threading, multiprocessing, platform,time

platform = platform.system()

WIDTH, HEIGHT = 800, 600
FACE_COLOR = (255, 255, 255)
DEBUG_MODE = True
USE_TEXT_QUERY = False
ONLINE_RAG = True
SERVOS = True
SERVO_ANGLES_DEBUG_TEXT_FREQUENCY = 120
IDLE_DIALOG_TIME = 90
OUTPUT_DEVICE = "bcm2835 Headphones, bcm2835 Headphones"
MICROPHONE_DEVICE = "Poly Blackwire 3320 Series: USB Audio"
CHAT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
AUDIO_RECOGNITION_MODEL = "openai/whisper-large-v3-turbo"
RAG_MODEL_ONLINE = "intfloat/multilingual-e5-large-instruct"
RAG_MODEL_LOCAL = ""
TTS_MODEL_PATH = "/voices/es_MX-claude-high"
HF_TOKEN = "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"

directorio = os.path.dirname(os.path.realpath(__file__))

def query_rag(habla, key_pressed, lock, lock2, lock3, lock4, lock5, query_text, state, generated_audio_index, reproduced_audio_index, commands_queue_index, i_amount):
    import random
    from huggingface_hub import InferenceClient
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from pygame import mixer

    client = InferenceClient(model=CHAT_MODEL, token=HF_TOKEN)
    stt_client = InferenceClient(model=AUDIO_RECOGNITION_MODEL, token=HF_TOKEN)

    servo_angles = {"rs": 0, "re": 90, "ls": 0, "le": 90}

    if (platform == "Linux" and SERVOS):
        print("Linux Platform Detected")
        print("Make sure the Servos are connected properly to the Raspberry Pi, for Orange Pi you will need to change wiringpi library")
        print("If you are using Linux OS on a VM or physical computer, you'll see an error with the wiringpi library most likely")
        print("Servo Signal Activated, the servos will move to AI commands")

        hilo_servo = threading.Thread(target=control_servo_movement, daemon=True, args=(servo_angles,))
        hilo_servo.start()

    if (DEBUG_MODE):
        print("Loading Embedding Model Connection")

    if (ONLINE_RAG):
        embedding_function = HuggingFaceInferenceAPIEmbeddings(model_name=RAG_MODEL_ONLINE, api_key=HF_TOKEN)
    else:
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embedding_function = HuggingFaceEmbeddings(model_name=RAG_MODEL_LOCAL, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs, api_key=HF_TOKEN)

    print("Initiating Embedding Test")

    try:
        print(embedding_function.embed_query("hello world")[:10])
        print("Test Successful!")
    except Exception as e:
        print("Embedding Test Errored: " + str(e))
        print("Check if it's an internet connection problem, this affects the performace of the program critically")

    CHROMA_PATH = directorio + "/Base de Datos Vectorial Chroma"

    PROMPT_TEMPLATE = """
Responde la anterior pregunta basado solamente en el siguiente contexto:

{context}

---

Responde la anterior pregunta basada en el contexto de arriba: {question}
"""
    
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    conversation = [{"role": "system", "content":
"""Your name is Luminia and you speak only in Spanish, you're an assistant for helping
students in the university with the information provided to you, giving short and preccise answers.

In your answers you must add commands between brackets to move the servo motors to give a more vibid look when
talking, you must put a number between 0 and 180 to determinate the rotation of the motor, you only can use them
once per sentence and they must be at the beginning of each sentence, they control parts of two arms, these commands are as follows:

[rs] - Moves the right shoulder, 0 is down, 180 is straight up.
[re] - Moves the right elbow, 90 is straight with the right shoulder orientation, 0 moves it to the center of the body
(must have right shoulder at 55 or higher), 180 moves it away from the body.
[ls] - Moves the left shoulder, 0 is down, 180 is straight up.
[le] - Moves the left elbow, 90 is straight with the left shoulder orientation, 0 moves it closer to the body
(must have left shoulder at 55 or higher), 180 moves it away from the body.

A sentence is counted when a dot (.), question mark, exclamation mark or semicolon is found in the response.

Here's an example of a response:
"[rs:135][re:135]Hola estudiante, muy buenos días.
[rs:45][re:135][ls:45][le:135]¿Hay algo en lo que pueda ayudarte hoy?
[ls:90]Puedes preguntarme lo que sea."

When someone asks for information about something or the location of some departament or places, if possible assume it's information from the
TecNM aka the Tecnologic Institute of Costa Grande, and respond only with: [RAG_SEARCH], nothing else, otherwise respond normally if no
question is asked.

After the proper context is given to you, try to answer the request from the student, if none of the information provided to you can
answer the question simply say that the information couldn't be found, maybe it's not in the database and probably should ask a person from the
institute instead.

In your responses avoid at all cost placing </s> or unreadable stuff without proper context, your responses are read aloud by a text to speech
application, so avoid placing any unreadable stuff in your response other than the commands allowed, additionally, mathematical expressions
such as x^2 put them as x squared and such so the text to speech can pronounce them properly."""},
    {'role': 'assistant', 'content': '[rs:135][re:135]Hola estudiante, ¿En que puedo servirte el dia de hoy?'}]
    presentation_dialog = [{"role": "system", "content": conversation[0]["content"]},
                           {'role': 'assistant', 'content': '[rs:135][re:135]Hola profesor, ¿En que puedo ser de utilidad el dia de hoy?'}]
    
    idle_timer = 0

    lock3.acquire()
    state.value = 0
    lock3.release()

    if (DEBUG_MODE):
        print("Voice Engine Initializing")
    
    if (platform == "Linux"):
        from dimits import Dimits
            
        mixer.init(devicename=OUTPUT_DEVICE)

        hilo_reproduce = threading.Thread(target=reproduce_audio, daemon=True, args=(lock, lock3, lock5, reproduced_audio_index, generated_audio_index, mixer, servo_angles, commands_queue_index, i_amount))
        hilo_reproduce.start()

        engine = Dimits(directorio + TTS_MODEL_PATH)
    else:
        import pyttsx3

        engine = pyttsx3.init()
            
        engine.setProperty("rate", 160)

    lock4.acquire()
    key_pressed.value = 0
    lock4.release()

    lock2.acquire()
    query_text.value = ""
    lock2.release()

    print("Ready to operate!")
    
    if (USE_TEXT_QUERY):
        print("Type something to the AI now")
        
    while True:
        if (idle_timer >= IDLE_DIALOG_TIME*20):
            idle_timer = 0

        if (USE_TEXT_QUERY):
            lock2.acquire()
            if (query_text.value == ""):
                lock2.release()
                time.sleep(0.05)
                
                continue
            
            lock2.release()
        else:
            lock4.acquire()
            key = key_pressed.value
            lock4.release()

            if (key == 0):
                idle_timer += 1
                
                if (idle_timer < IDLE_DIALOG_TIME*20):
                    time.sleep(0.05)

                    continue
            else:
                idle_timer = 0
            
            if (DEBUG_MODE):
                if (idle_timer < IDLE_DIALOG_TIME*20):
                    print("Key Detected")
                else:
                    print("Initiaing Idle Dialog")
            
            lock2.acquire()
            if (idle_timer < IDLE_DIALOG_TIME*20):
                try:
                    query_text.value = listen_microphone(stt_client, lock3, lock4, state, key_pressed)
                except Exception as e:
                    print("STT errored: " + str(e))
                    
                    query_text.value = ""
            else:
                query_text.value = "Presentaté ante los estudiantes universitarios por favor."

                lock3.acquire()
                state.value = 2
                lock3.release()
            
            if (query_text.value == "" or query_text.value == "Error, did not understand what you said."):
                lock2.release()

                lock4.acquire()
                key_pressed.value = 0
                lock4.release()

                lock3.acquire()
                state.value = 0
                lock3.release()

                time.sleep(0.05)

                continue

            lock2.release()

        if (DEBUG_MODE):
            print("User Text Received: " + query_text.value)

        do_rag = False
        loop = True

        while (loop):
            if (idle_timer < IDLE_DIALOG_TIME*20 and do_rag):
                if (DEBUG_MODE):
                    print("Searching Chroma Database Similarity")
                    
                # Search the DB.
                while True:
                    try:
                        results = db.similarity_search_with_score(query_text.value, k=10)
                        break
                    except:
                        pass

                if (DEBUG_MODE):
                    print("Formatting User Prompt")

                context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
                prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text.value)
                # print(prompt)
            else:
                prompt = query_text.value

            if (idle_timer < 1000):
                conversation.append({"role": "user", "content": prompt})
            else:
                presentation_dialog.append({"role": "user", "content": prompt})

            if (DEBUG_MODE):
                print("Generating AI Response")

            lock3.acquire()
            state.value = 2
            lock3.release()

            while True:
                try:
                    if (idle_timer < IDLE_DIALOG_TIME*20):
                        completion = client.chat_completion(conversation, stream=True, max_tokens=1000, seed=random.randint(10000000,100000000))
                    else:
                        completion = client.chat_completion(presentation_dialog, stream=True, max_tokens=1000, seed=random.randint(10000000,100000000))
                    break
                except:
                    pass
            
            if (DEBUG_MODE):
                print("AI Reponse Generating.")

            response_text = ""
            temp_msg = ""
            phrase_enders = [".", "?", "!", ";", "]"]
            
            lock5.acquire()
            generated_audio_index.value = 1
            reproduced_audio_index.value = 1
            i_amount.value = 1000
            commands_queue_index[:] = []
            lock5.release()
            
            try:
                for chunk in completion:
                    if (chunk.choices[0].delta.content is not None):
                        part = chunk.choices[0].delta.content
                        response_text += part
                        temp_msg += part
                        index = len(temp_msg)
                            
                        for character in phrase_enders:
                            index2 = temp_msg.find(character)
                            if (index2 >= 0):
                                index = min(index, index2)

                        if (index < len(temp_msg)):
                            do_rag = speak(habla, lock, lock3, lock5, temp_msg[:index + 1], servo_angles, engine, generated_audio_index, commands_queue_index)
                            
                            if (do_rag):
                                break

                            if (platform == "Linux"):
                                lock5.acquire()
                                generated_audio_index.value += 1
                                lock5.release()
                            
                            temp_msg = temp_msg[index + 1:]
                            
                        #print(part, end="")
                        #sys.stdout.flush()
                
                if (do_rag):
                    lock3.acquire()
                    state.value = 3
                    lock3.release()

                    continue
            except Exception as e:
                if (DEBUG_MODE):
                    print("AI Generation Streaming Errored: " + str(e))
            
            if (DEBUG_MODE):
                print("AI Generated: " + response_text)
            
            if ("[RAG_SEARCH]" in temp_msg):
                do_rag = True
                continue

            loop = False

        if (idle_timer < IDLE_DIALOG_TIME*20):
            if (do_rag):
                conversation.pop()
                conversation.append({"role": "user", "content": query_text.value})
            
            conversation.append({"role": "assistant", "content": response_text})
        else:
            presentation_dialog.pop()

        if (DEBUG_MODE and do_rag):
            sources = [doc.metadata.get("id", None) for doc, _score in results]
            print(f"Context Sources: {sources}")
        
        lock4.acquire()
        key_pressed.value = 0
        lock4.release()

        lock2.acquire()
        query_text.value = ""
        lock2.release()
        
        while True:
            lock5.acquire()
            if (i_amount.value == 1000):
                i_amount.value = generated_audio_index.value
            elif (i_amount.value <= reproduced_audio_index.value):
                break
            lock5.release()

            time.sleep(0.05)
        lock5.release()

        print("Ready to operate again!")

        if (USE_TEXT_QUERY):
            print("Type something to the AI now")

def reproduce_audio(lock, lock3, lock5, j, i, mixer, servo_angles, i_list, i_amount):
    while True:
        lock5.acquire()
        j_value = j.value
        i_value = i.value
        i_amount_value = i_amount.value
        lock5.release()
        
        if (j_value >= i_value):
            if (j_value >= i_amount_value):
                servo_angles["rs"] = 0
                servo_angles["re"] = 90
                servo_angles["ls"] = 0
                servo_angles["le"] = 90
            
            time.sleep(0.05)
            
            continue
        index = j_value - 1

        servo_angles["rs"] = i_list[index]["rs"]
        servo_angles["re"] = i_list[index]["re"]
        servo_angles["ls"] = i_list[index]["ls"]
        servo_angles["le"] = i_list[index]["le"]

        mixer.music.load(directorio + "/audios/speech" + str(index + 1) + ".wav")
        mixer.music.play()

        lock3.acquire()
        state.value = 0
        lock3.release()

        lock.acquire()
        habla.value = 1
        lock.release()

        while True:
            if (not mixer.music.get_busy()):
                break
                                
            time.sleep(0.05)
                            
        lock.acquire()
        habla.value = 0
        lock.release()
        
        lock5.acquire()
        j.value += 1
        lock5.release()

def speak(habla, lock, lock3, lock5, text, servo_angles, engine, i, i_list):
    if (DEBUG_MODE):
        print("Raw TTS: " + text)
    
    if (platform == "Linux"):
        lock5.acquire()
        index = len(i_list[:]) - 1
        if (index == -1):
            i_list.append({"rs": 0, "re": 90, "ls": 0, "le": 90})
        else:
            i_list.append({"rs": i_list[index]["rs"], "re": i_list[index]["re"], "ls": i_list[index]["ls"], "le": i_list[index]["le"]})
        lock5.release()

    index = text.find("[")
    while (index >= 0):
        index2 = text.find("]")
        
        if (index2 >= 0):
            result = execute_command(lock5, text[index + 1:index2], servo_angles, i_list)

            if (result == "RAG"):
                return True

            text = text[:index] + text[index2 + 1:]
        index = text.find("[")

    if (DEBUG_MODE):
        print("Speaking TTS: " + text)
        
    if (text == "" or text == "."):
        lock5.acquire()
        i.value -= 1
        lock5.release()
        
        return False

    if (platform == "Linux"):
        lock5.acquire()
        aux = i.value
        lock5.release()
                
        engine.text_2_audio_file(text, "audios/speech" + str(aux), directorio, format="wav")
    else:
        engine.say(text)

        lock3.acquire()
        state.value = 0
        lock3.release()

        lock.acquire()
        habla.value = 1
        lock.release()

        engine.runAndWait()

        lock.acquire()
        habla.value = 0
        lock.release()
    
    return False

def listen_microphone(client, lock3, lock4, state, key_pressed):
    import speech_recognition as sr, pyaudio, wave, struct
    
    AUDIO_FILE = directorio + "/audios/recording.wav"
    SILENCE_TIME = 2
    LISTENING_TIME = 10
    audio_buffer = []

    if (DEBUG_MODE):
        print("Trying to open microphone")
    
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    for i in range(0, device_count):
        device_info = p.get_device_info_by_index(i)
        if (MICROPHONE_DEVICE in device_info["name"]):
            device_index = i
            
            break
    
    FORMAT = pyaudio.paInt16
    CHANNELS = device_info["maxInputChannels"]
    RATE = int(device_info["defaultSampleRate"])
    CHUNK = 1024
    
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=device_index, frames_per_buffer=CHUNK)
    silence_start = None
    started_talking = False
    listening_start = time.time()

    if (DEBUG_MODE):
        print("Listening...")

    lock3.acquire()
    state.value = 1
    lock3.release()

    while True:
        data = stream.read(CHUNK)
        audio_buffer.append(data)

        if (len(audio_buffer) > 0):
            data = audio_buffer[-1]
            samples = struct.unpack(f'{len(data)//2}h', data)  # "h" para enteros de 16 bits (paInt16)
            volume = max(abs(sample) for sample in samples)

            if (volume < 500 and started_talking):
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_TIME:
                    if (DEBUG_MODE):
                        print("User Stopped Talking.")

                    break
            elif (not started_talking):
                started_talking = True
            else:
                silence_start = None
        
        if (time.time() - listening_start >= LISTENING_TIME):
            if (DEBUG_MODE):
                print("Timeout.")

            break

        lock4.acquire()
        key = key_pressed.value
        lock4.release()

        if (key == 2):
            if (DEBUG_MODE):
                print("Process Stopped.")
            
            break

    stream.stop_stream()
    stream.close()
    p.terminate()
        
    if (DEBUG_MODE):
        print("Stopped listening.")

    wf = wave.open(AUDIO_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(audio_buffer))
    wf.close()

    recognizer = sr.Recognizer()
    with sr.AudioFile(directorio + "/audios/recording.wav") as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source)
    
    lock3.acquire()
    state.value = 2
    lock3.release()

    try:
        return client.automatic_speech_recognition(audio.get_wav_data()).text
    except Exception as e:
        print("Recognizer STT Errored: ",end="")
        print(e)

        return "Error, did not understand what you said."

def execute_command(lock5, text, servo_angles, i_list):
    separation = text.split(":")
    command = separation[0].replace(" ","")
    value = int(separation[1])

    if (DEBUG_MODE):
        print("Command Executed: " + command + " with value " + str(value))

    if (value < 0 or value > 180):
        return

    if (command == "RAG_SEARCH"):
        return "RAG"
    
    if (platform == "Linux"):
        lock5.acquire()
        index = len(i_list[:]) - 1
        servo_angles = i_list[index]
        lock5.release()

    if (command == "rs"):
        servo_angles["rs"] = value

        if (2*value + servo_angles["re"] < 90):
            servo_angles["re"] = 90 - 2*value
        elif (2*value + servo_angles["re"] > 450):
            servo_angles["re"] = 450 - 2*value
    elif (command == "re"):
        servo_angles["re"] = value
            
        if (value + 2*servo_angles["rs"] < 90):
            servo_angles["rs"] = (90 - value)/2
        elif (2*value + servo_angles["rs"] > 450):
            servo_angles["rs"] = (450 - value)/2
    elif (command == "ls"):
        servo_angles["ls"] = value
            
        if (2*value + servo_angles["le"] < 90):
            servo_angles["le"] = 90 - 2*value
        elif (2*value + servo_angles["le"] > 450):
            servo_angles["le"] = 450 - 2*value
    elif (command == "le"):
        servo_angles["le"] = value

        if (value + 2*servo_angles["ls"] < 90):
            servo_angles["ls"] = (90 - value)/2
        elif (2*value + servo_angles["ls"] > 450):
            servo_angles["ls"] = (450 - value)/2

    if (platform == "Linux"):
        lock5.acquire()
        i_list[index] = servo_angles
        lock5.release()
    
    return ""

def control_servo_movement(servo_angles):
    try:
        import pigpio

        if (DEBUG_MODE):
            debug_timer = 0

        rs_pin = 22
        re_pin = 23
        ls_pin = 24
        le_pin = 25

        rs_angle_cur = 0
        re_angle_cur = 90
        ls_angle_cur = 0
        le_angle_cur = 90

        pi = pigpio.pi()

        pi.set_mode(rs_pin, pigpio.OUTPUT)
        pi.set_mode(re_pin, pigpio.OUTPUT)
        pi.set_mode(ls_pin, pigpio.OUTPUT)
        pi.set_mode(le_pin, pigpio.OUTPUT)
        
        pi.set_servo_pulsewidth(rs_pin, 2500)
        pi.set_servo_pulsewidth(re_pin, 1500)
        pi.set_servo_pulsewidth(ls_pin, 2500)
        pi.set_servo_pulsewidth(le_pin, 1500)
        
        if (DEBUG_MODE):
            print("Servo Loop Started")
        
        while True:
            rs_angle_cur = rs_angle_cur + (servo_angles["rs"] - rs_angle_cur)/25
            re_angle_cur = re_angle_cur + (servo_angles["re"] - re_angle_cur)/25
            ls_angle_cur = ls_angle_cur + (servo_angles["ls"] - ls_angle_cur)/25
            le_angle_cur = le_angle_cur + (servo_angles["le"] - le_angle_cur)/25
            
            if (DEBUG_MODE):
                if (debug_timer == 0):
                    print("Servo Right Shoulder Angle: " + str(rs_angle_cur))
                    print("Servo Right Elbow Angle: " + str(re_angle_cur))
                    print("Servo Left Shoulder Angle: " + str(ls_angle_cur))
                    print("Servo Left Elbow Angle: " + str(le_angle_cur))

                    debug_timer = SERVO_ANGLES_DEBUG_TEXT_FREQUENCY*20
                else:
                    debug_timer -= 1

            pi.set_servo_pulsewidth(rs_pin, 500 + 2000*(180 - rs_angle_cur)/180)
            pi.set_servo_pulsewidth(re_pin, 500 + 2000*(180 - re_angle_cur)/180)
            pi.set_servo_pulsewidth(ls_pin, 500 + 2000*(180 - ls_angle_cur)/180)
            pi.set_servo_pulsewidth(le_pin, 500 + 2000*(180 - le_angle_cur)/180)

            time.sleep(0.05)
    except Exception as e:
        print("Servo Manager Errored: " + str(e))
        print("Servo Signal Deactivated, the servos won't move anymore")

def leer_input(query, lock2):
    while True:
        leido = input("")

        lock2.acquire()
        if (leido != "" and query.value == ""):
            query.value = leido
        lock2.release()

if (__name__ == "__main__"):
    import multiprocessing, pygame, subprocess

    manager = multiprocessing.Manager()
    lock = multiprocessing.Lock()
    lock2 = multiprocessing.Lock()
    lock3 = multiprocessing.Lock()
    lock4 = multiprocessing.Lock()
    lock5 = multiprocessing.Lock()
    habla = manager.Value("d", 0)
    query = manager.Value("s", "")
    state = manager.Value("d", 2) #0 is face, 1 is listening, 2 is processing, 3 is searching
    amount_audios = manager.Value("d", 0)
    key_pressed = manager.Value("d", 0)
    generated_audio_index = manager.Value("d", 0)
    reproduced_audio_index = manager.Value("d", 0)
    commands_queue_index = manager.list()
    
    proceso_ai = multiprocessing.Process(target=query_rag, daemon=True, args=(habla, key_pressed, lock, lock2, lock3, lock4, lock5, query, state, generated_audio_index, reproduced_audio_index, commands_queue_index, amount_audios))
    proceso_ai.start()

    if (USE_TEXT_QUERY):
        hilo_input = threading.Thread(target=leer_input, daemon=True, args=(query, lock2))
        hilo_input.start()

    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED)
    pygame.display.set_caption("Asistente Luminia")
    pygame.mouse.set_visible(False)

    listening_image = pygame.image.load(os.path.dirname(os.path.realpath(__file__)) + "/imagenes/listening.png").convert_alpha()
    processing_image = pygame.image.load(os.path.dirname(os.path.realpath(__file__)) + "/imagenes/processing.png").convert_alpha()
    searching_image = pygame.image.load(os.path.dirname(os.path.realpath(__file__)) + "/imagenes/searching.png").convert_alpha() 
    background_image = pygame.image.load(os.path.dirname(os.path.realpath(__file__)) + "/imagenes/bg.png").convert_alpha()
    bg_width, bg_height = background_image.get_size()

    # Reloj para controlar la velocidad de fotogramas
    clock = pygame.time.Clock()

    def draw_rotated_face(center_x, center_y, angle):
        face_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

        draw_rotated_eye(face_surface, WIDTH // 2 + (face_leftX - 60*face_scale), HEIGHT // 2 - (face_leftY + 50*face_scale), face_leftSizeX*face_scale, face_leftSizeY*face_scale, face_leftAngle)
        draw_rotated_eye(face_surface, WIDTH // 2 + (face_rightX + 60*face_scale), HEIGHT // 2 - (face_rightY + 50*face_scale), face_rightSizeX*face_scale, face_rightSizeY*face_scale, face_rightAngle)
        draw_rotated_rect(face_surface, WIDTH // 2 + face_mouthX, HEIGHT // 2 - (face_mouthY - (60 + face_mouthSizeY/2)*face_scale), face_mouthSizeX*face_scale, (10 + face_mouthSizeY)*face_scale, face_mouthAngle)

        rotated_face = pygame.transform.rotate(face_surface, angle)
        face_rect = rotated_face.get_rect(center=(center_x, center_y))

        screen.blit(rotated_face, face_rect)

    # Función para dibujar y rotar ojos
    def draw_rotated_eye(surface, center_x, center_y, width, height, angle):
        # Crear superficie para el ojo
        eye_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.ellipse(eye_surface, FACE_COLOR, (0, 0, width, height))
        
        # Rotar superficie
        rotated_eye = pygame.transform.rotate(eye_surface, angle)
        eye_rect = rotated_eye.get_rect(center=(center_x, center_y))
        
        # Dibujar ojo rotado en la pantalla
        surface.blit(rotated_eye, eye_rect)

    def draw_rotated_rect(surface, center_x, center_y, width, height, angle):
        # Crear superficie para el ojo
        rect_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(rect_surface, FACE_COLOR, (0, 0, width, height))
        
        # Rotar superficie
        rotated_eye = pygame.transform.rotate(rect_surface, angle)
        eye_rect = rotated_eye.get_rect(center=(center_x, center_y))
        
        # Dibujar ojo rotado en la pantalla
        surface.blit(rotated_eye, eye_rect)

    def draw_background():
        bg_surface = pygame.Surface((WIDTH + 200, HEIGHT + 300), pygame.SRCALPHA)

        for x in range(-bg_width, WIDTH + 200, bg_width):
            for y in range(-bg_height, HEIGHT + 300, bg_height):
                bg_surface.blit(background_image, (x + offset_x, y + offset_y))
        
        rotated_bg = pygame.transform.rotate(bg_surface, -20)
        bg_rect = rotated_bg.get_rect(center=(420, 300))

        screen.blit(rotated_bg, bg_rect)

    timer = 0
    timer2 = 0
    offset_x = 0
    offset_y = 0

    running = True
    checked = False
    last_time = None
    last_time2 = None
    shutdown = True
    fullscreen = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                shutdown = False
                running = False
            
            if (event.type == pygame.KEYDOWN):
                lock4.acquire()
                key = key_pressed.value
                lock4.release()
                
                if (event.key == pygame.K_q):
                    shutdown = False
                    running = False
                elif (event.key == pygame.K_r):
                    if (key < 2):
                        checked = True

                        lock4.acquire()
                        key_pressed.value += 1
                        lock4.release()
                    
                    if (last_time2 is not None and time.time() - last_time2 <= 1):
                        print("YES")
                        running = False
                        
                        break

                    if (last_time is None or time.time() - last_time > 1):
                        last_time = time.time()
                    elif (time.time() - last_time <= 1):
                        last_time2 = time.time()
                elif (checked):
                    checked = False
            else:
                pressed_time = None

                if (checked):
                    checked = False

        if (not fullscreen):
            try:
                pygame.display.toggle_fullscreen()
                fullscreen = True
            except:
                pass

        offset_x = (offset_x + 3) % bg_width
        offset_y = (offset_y + 1.5) % bg_height

        draw_background()

        timer += 1.5
        
        lock.acquire()
        if (habla.value == 1 or timer2 > 0):
            timer2 += 1.5
            if (timer2%24 == 0):
                timer2 = 0
        lock.release()

        eye_width, eye_height = 60, 90
        face_scale = 2
        face_angle = 5*math.sin(math.radians(0.5*timer))
        face_leftX = 5*math.sin(math.radians(1.3*timer))
        face_leftY = 5*math.sin(math.radians(2*timer))
        face_leftAngle = 7*math.sin(math.radians(0.76*timer))
        face_leftSizeX = 90
        face_leftSizeY = 65 + 55*math.cos(math.radians(360*min(max(timer%800, 0), 15))/15)
        face_rightX = -5*math.sin(math.radians(1.1*timer))
        face_rightY = 5*math.sin(math.radians(1.76*timer))
        face_rightAngle = -7*math.sin(math.radians(0.93*timer))
        face_rightSizeX = 90
        face_rightSizeY = 65 + 55*math.cos(math.radians(360*min(max(timer%800, 0), 15))/15)
        face_mouthX = 3*math.sin(math.radians(0.73*timer))
        face_mouthY = -4*math.sin(math.radians(0.94*timer))
        face_mouthAngle = -5*math.sin(math.radians(1.2*timer))
        face_mouthSizeX = 180
        face_mouthSizeY = 30 - 15*math.cos(math.radians(15*timer2))
        
        lock3.acquire()
        if (state.value == 0):
            draw_rotated_face(WIDTH // 2, HEIGHT // 2, face_angle)
        elif (state.value == 1):
            timer = 0
            screen.blit(listening_image, (0, 0, WIDTH, HEIGHT))
        elif (state.value == 2):
            timer = 0
            screen.blit(processing_image, (0, 0, WIDTH, HEIGHT))
        else:
            timer = 0
            screen.blit(searching_image, (0, 0, WIDTH, HEIGHT))
        lock3.release()

        # Actualizar pantalla
        pygame.display.flip()

        # Controlar FPS
        clock.tick(60)

    # Cerrar Pygame
    pygame.quit()

    if (platform == "Linux" and shutdown):
        subprocess.call(["shutdown", "-h", "now"])
