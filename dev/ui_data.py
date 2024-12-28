import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QMainWindow,
)


class CameraStream(QWidget):
    def __init__(self):
        super().__init__()
        self.video_label = QLabel(self)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.video_label)
        self.setLayout(self.layout)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


class TrafficLight(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        self.red_light = QLabel(self)
        self.yellow_light = QLabel(self)
        self.green_light = QLabel(self)

        self.setup_light(self.red_light, "red")
        self.setup_light(self.yellow_light, "yellow")
        self.setup_light(self.green_light, "green")

        self.layout.addWidget(self.red_light)
        self.layout.addWidget(self.yellow_light)
        self.layout.addWidget(self.green_light)
        self.setLayout(self.layout)

    def setup_light(self, label, color):
        label.setFixedSize(300, 300)
        label.setStyleSheet(f"background-color: {color}; border-radius: 125px;")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)

        self.camera_stream = CameraStream()
        self.traffic_light = TrafficLight()

        self.layout.addWidget(self.camera_stream)
        self.layout.addWidget(self.traffic_light)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
