import time
import pyfirmata
from pyfirmata import Arduino
import serial.tools.list_ports

# Define the digital pins for LEDs
LED_PIN_1 = 5
LED_PIN_2 = 6

# Initialize Arduino board

# Define function to initialize Arduino board and stepper motors

all_comports = serial.tools.list_ports.comports()
com_ports = []

# Find Arduino Mega's COM port
for comport in all_comports:
    if str(comport.description).startswith("USB-SERIAL"):
        com_ports.append(str(comport.device))

if not com_ports:
    print("Arduino Mega not found.")
    

print("Arduino Mega found at COM ports:", com_ports)

try:
    boards = []
    for com_port in com_ports:
        board = pyfirmata.Arduino(com_port)
        boards.append(board)
except:
    pass

# Define the pins as output
board.digital[LED_PIN_1].mode = pyfirmata.OUTPUT

board.digital[LED_PIN_2].mode = pyfirmata.OUTPUT


# board.digital[LED_PIN_1].write(1)
# board.digital[LED_PIN_2].write(0)

board.digital[LED_PIN_1].write(0)
board.digital[LED_PIN_2].write(1)



