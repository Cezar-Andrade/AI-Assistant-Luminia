import pigpio, time

pin = 22

pi = pigpio.pi()
zip
pi.set_mode(pin, pigpio.OUTPUT)

while True:
	pi.set_servo_pulsewidth(pin, 500)
	time.sleep(1)
	pi.set_servo_pulsewidth(pin, 1000)
	time.sleep(1)
	pi.set_servo_pulsewidth(pin, 1500)
	time.sleep(1)
	#pin += 1
