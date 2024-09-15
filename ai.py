from ultralytics import YOLO


class DetectionModel:
    def __init__(self):
        self.model = YOLO("food-detection.pt")

    def predict(self, img):
        return self.model.predict(img)


class TrayDetectionModel:
    def __init__(self):
        self.model = YOLO("tray-detection.pt")

    def predict(self, img):
        return self.model.predict(img)


class AmountModel:
    pass
