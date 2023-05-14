import os

from ultralytics import YOLO


class Method:
    experts: list[YOLO] = []

    def __init__(self, path: str, ext: tuple[str] = ('.pt')):
        if os.path.isfile(path):
            self.experts += [YOLO(path)]
            return

        for fname in os.listdir(path):
            if not fname.endswith(ext):
                continue

            f = os.path.join(path, fname)
            if not os.path.isfile(f):
                continue

            print(f)
            self.experts += [YOLO(f)]

    def predict(self, input: str, output: str):
        pass
