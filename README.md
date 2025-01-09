
# Zebra-Crossing Person Detection and Operating Traffic Signal

## Overview
This project focuses on detecting persons and mobile phone usage at zebra crossings using advanced computer vision techniques. The system integrates person detection, mobile phone detection, and traffic signal operation to improve road safety and enforce rules effectively.

## Features
- **Person Detection:** Uses YOLOv8 model to detect persons.
- **Mobile Phone Detection:** Identifies individuals using mobile phones while crossing the zebra.
- **Traffic Signal Control:** Automatically switches traffic signals based on detection.
- **Audio Notifications:** Alerts people when a person or mobile phone is detected using a text-to-speech system.
- **Arduino Integration:** Operates LED lights to indicate signal status.
- **Frame Storage:** Saves frames with detected mobile phone usage for future analysis.
- **Real-Time Processing:** Handles video input from RTSP cameras in real time.

## Project Components
### 1. Hardware
- **Arduino Board (COM4):** Controls LEDs for signaling.
- **LEDs:** Two LEDs for traffic signal status.
- **RTSP Camera:** Captures live video for analysis.

### 2. Software
- **Programming Languages:** Python
- **Frameworks and Libraries:**
  - `cv2` for video processing
  - `ultralytics` for YOLOv8 models
  - `pyfirmata` for Arduino communication
  - `pyModbusTCP` for Modbus connection
  - `pyttsx3` for text-to-speech
- **Models Used:**
  - YOLOv8 model for person detection
  - YOLOv8 model for mobile phone detection

### 3. Functional Workflow
1. Captures video stream via RTSP.
2. Defines a polygonal region for detection.
3. Detects persons and checks if they are inside the defined region.
4. If a person is detected, the system:
   - Activates the red signal LED.
   - Verifies if the person is using a mobile phone.
   - Provides audio alerts for violations.
5. Stores frames with detected violations for record-keeping.

## Installation
### Prerequisites
1. Python 3.8+
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Arduino IDE to configure the board.

### Clone the Repository
```bash
git clone https://github.com/<your-github-username>/zebra-crossing-person-detection-and-operating-Traffic-Signal.git
cd zebra-crossing-person-detection-and-operating-Traffic-Signal
```

### Set Up the Environment
1. Ensure YOLO models are in the `assets/yolov8_models/` directory.
2. Specify the Arduino COM port in the code (`COM4` in this case).
3. Configure RTSP camera link in `stream1` function.
4. Create a directory to save detected frames:
   ```bash
   mkdir detected_frames
   ```

## How to Run
1. Connect the Arduino board and ensure the LEDs are properly configured.
2. Run the main script:
   ```bash
   python main.py
   ```
3. Monitor the system via the displayed video stream window.
4. Press `q` to stop the application.

## Directory Structure
```
.
├── assets
│   └── yolov8_models
│       ├── person_model.pt
│       ├── best.pt
├── detected_frames
├── main.py
├── requirements.txt
└── README.md
```

## Future Enhancements
- Add support for multiple camera inputs.
- Implement cloud integration for real-time monitoring and analytics.
- Enhance detection accuracy with custom-trained models.

## Contributing
Feel free to open an issue or submit a pull request for improvements.

## License
This project is licensed under the MIT License.

## Demo Link
https://drive.google.com/file/d/1Xznft5WLPIkclWYFkzZNoBNDTotuFQcz/view?usp=sharing
