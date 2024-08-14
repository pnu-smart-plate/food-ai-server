from ultralytics import YOLO


class DetectionModel:
    def __init__(self):
        self.model = YOLO("best.pt")

    def predict(self, img):
        return self.model.predict(img)


class AmountModel:
    pass
