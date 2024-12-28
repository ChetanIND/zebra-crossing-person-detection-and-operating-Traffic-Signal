import cv2
import threading
import queue
import numpy as np
import time
from ultralytics import YOLO
import pandas as pd
from pyModbusTCP.client import ModbusClient
import socket
import pyttsx3
import torch
import time
import pyfirmata
from pyfirmata import Arduino
import serial.tools.list_ports
import os 

# Define the digital pins for LEDs
LED_PIN_1 = 5
LED_PIN_2 = 6

# ==========================================================================================
# LED ARDUINO CONNECTION
# ==========================================================================================

board = pyfirmata.Arduino("COM4")

# Lock for pyttsx3 runAndWait()
engine_lock = threading.Lock()

# ==========================================================================================
# MODBUS CONNECTION & text-to-audio object initialization
# ==========================================================================================

try:
    engine = pyttsx3.init()
    hostname = socket.gethostname()
    server_ip_address = socket.gethostbyname(hostname)
    server_port = 502
    client = ModbusClient(server_ip_address, server_port)
except Exception as e:
    print(f"Error initializing MODBUS or TTS: {e}")
    pass


# ==========================================================================================
# MODEL PATH DECLARATION & INITIALIZATION
# ==========================================================================================
# Making the code device-agnostic
device = 'cuda' if torch.cuda.is_available() else 'cpu'

person_model_path = r"assest\yolov8_models\person_model.pt"
mobile_model_path = r"assest\yolov8_models\best.pt"

person_model = YOLO(person_model_path).to(device)  # Move person detection model to GPU
mobile_model = YOLO(mobile_model_path).to(device)  # Move mobile detection model to GPU

# Create a directory for saving detected frames
detected_frames_dir = r"C:\Users\mukesh\Desktop\TATA\detected_frames"
os.makedirs(detected_frames_dir, exist_ok=True)

# ==========================================================================================
# ORIGINAL VIDEOCAPTURE CLASS
# ==========================================================================================

class VideoCapture:

    def __init__(self, camera_link):
        self.cap = cv2.VideoCapture()
        self.cap.open(camera_link)

        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

def person_detected():
    with engine_lock:
        engine.say("Person detected")
        engine.runAndWait()

def mobile_detected():
    with engine_lock:
        engine.say("Don't use the mobile")
        engine.runAndWait()

def tts_task(message):
    if message == "Person detected":
        person_detected()

    if message == "Don't use the mobile":
        mobile_detected()
        print("Mobile Detected")

# STREAM 1
def stream1():

    # camera link
    camera_link = r"rtsp://admin:Mk123456@192.168.0.100:554/Streaming/Channels/101"

    # MAIN POLYGON VERTICES
    # ==========================================================================================
    vertices = np.array([(220,380),(250,650),(800, 120),(650, 100)], np.int32)
    vertices = vertices.reshape((-1, 1, 2))
    # ==========================================================================================

    # fps parameters
    start_time = time.time()
    display_time = 1
    fc = 0
    FPS = 0

    status_counter = 0
    mobile_detected_flag = False  # Flag to detect mobile and break the loop

    # ==========================================================================================
    cap = VideoCapture(camera_link=camera_link)

    while True:
        fc += 1
        TIME = time.time() - start_time

        # calculate time to calculate FPS
        if TIME >= display_time:
            FPS = fc / TIME
            start_time = time.time()

        # string formatting for FPS
        FPS = round(FPS)
        fps_disp = "FPS: " + str(FPS)[:5]
        frame = cap.read()
        frame = cv2.resize(frame, (740, 580))
        cv2.polylines(frame, [vertices], isClosed=True, color=(255, 255, 0), thickness=2)

        status_counter += 1
        # ====================================================================================
        # LIGHT INITIALIZE 
        # ====================================================================================

        all_comports = serial.tools.list_ports.comports()
        # board = connect_arduino(all_comports)
        board.digital[LED_PIN_1].mode = pyfirmata.OUTPUT
        board.digital[LED_PIN_2].mode = pyfirmata.OUTPUT

        # ====================================================================================
        person_results = person_model(frame, conf=0.7, verbose=False)
        person_inside_polygon = False

        # Process person detections
        if person_results and len(person_results[0].boxes) > 0:
            a = person_results[0].boxes.cpu().numpy().data
            px = pd.DataFrame(a).astype("float")
            number_of_objects = len(px)
            
            for result in person_results:
                for i in range(number_of_objects):
                    box = result.boxes.xywh[i].cpu().numpy()
                    person_x, person_y, person_width, person_height = map(int, box)

                    # Adjusting the y-coordinate to be at the feet
                    person_feet_y = person_y + person_height // 2

                    person_inside_polygon1 = cv2.pointPolygonTest(vertices, (person_x, person_feet_y), False)

                    # CHECK IF PERSON INSIDE POLYGON
                    if person_inside_polygon1 > 0:
                        person_inside_polygon = True
                        if status_counter > 5:
                            # tts_thread = threading.Thread(target=tts_task, args=("Person detected",))
                            # tts_thread.start()

                            board.digital[LED_PIN_1].write(0)
                            board.digital[LED_PIN_2].write(1)
                            
                            cv2.putText(frame, "RED SIGNAL", (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 3, cv2.LINE_AA)
                            status_counter = 0

                        cv2.rectangle(frame, (person_x - person_width // 2, person_y - person_height // 2), 
                                      (person_x + person_width // 2, person_y + person_height // 2), 
                                      (0, 255, 255), 2)
                        cv2.putText(frame, "P:inside", (person_x-30, person_y-30), 
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.circle(frame, (person_x, person_feet_y), 5, (0, 255, 255), -1)
                        cv2.circle(frame, (person_x, person_feet_y), 7, (0, 0, 0), 2)

                    else:
                        cv2.putText(frame, "P:outside", (person_x-30, person_y-20), 
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.circle(frame, (person_x, person_feet_y), 5, (0, 255, 0), -1)

        # If person is detected inside polygon, check for mobile detections
        if person_inside_polygon and not mobile_detected_flag:
            mobile_results = mobile_model(frame, conf=0.3, verbose=False)
            if mobile_results and len(mobile_results[0].boxes) > 0:
                a = mobile_results[0].boxes.cpu().numpy().data
                px = pd.DataFrame(a).astype("float")
                number_of_objects = len(px)
                
                for result in mobile_results:
                    for i in range(number_of_objects):
                        box = result.boxes.xywh[i].cpu().numpy()
                        mobile_x, mobile_y, mobile_width, mobile_height = map(int, box)
                        tts_thread = threading.Thread(target=tts_task, args=("Don't use the mobile",))
                        tts_thread.start()

                        # Save the frame
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        frame_filename = os.path.join(detected_frames_dir, f"mobile_detected_{timestamp}.jpg")
                        cv2.imwrite(frame_filename, frame)

                        mobile_detected_flag = True  # Set the flag to True to break the loop

        # LEFT SIDE INFORMATION
        cv2.putText(frame, fps_disp, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Muks Robotics", frame)
        cv2.imwrite("stream1.jpg", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# ==========================================================================================
#                                      MAIN CODE
# ==========================================================================================
if __name__ == "__main__":
    stream1()
