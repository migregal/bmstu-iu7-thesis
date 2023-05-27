
import os

from ultralytics import YOLO


class Train:
    model: YOLO
    path: str

    def __init__(self, path: str):
        self.model = YOLO("yolov8n.pt")
        self.path = path

    def train(self, imgsz: int = 640, epochs: int = 100, batch: int = 9):
        path = os.path.join(self.path, 'data.yaml')
        self.model.train(data=path, imgsz=imgsz, epochs=epochs, batch=batch, deterministic=False, cache='ram')
