import pyttsx3,time,os
from pygame import mixer, _sdl2

directorio = os.path.dirname(os.path.realpath(__file__))

engine = pyttsx3.init()

mixer.init()
print(_sdl2.audio.get_audio_device_names(False))
mixer.quit()
	
mixer.init(devicename = "bcm2835 Headphones, bcm2835 Headphones")

engine.setProperty("voice", "spanish")
engine.save_to_file("Hola papus, gracias por escucharme", directorio + "/speech.wav")
engine.runAndWait()

mixer.init(devicename="bcm2835 Headphones, bcm2835 Headphones");
mixer.music.load(directorio + "/speech.wav")
mixer.music.play()

while True:
	if (not mixer.music.get_busy()):
		break
	
	time.sleep(0.02)
