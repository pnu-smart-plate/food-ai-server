import json
import torch
from fastapi import FastAPI, UploadFile
import uvicorn
import ai

import io
from PIL import Image
import matplotlib.pyplot as plt

app = FastAPI()
detect_ai = ai.DetectionModel()
amount_ai = ai.AmountModel()


@app.post("/analyze")
async def detect(file: UploadFile):
    img = await file.read()

    # 모델을 실행하고 결과를 받아오는 부분
    predict = detect_ai.predict(img_convert(img))[0]

    # get_bounding_box(x)
    result_json = detection_to_json(predict.boxes)
    # x.show()
    return result_json


def img_show(image_bytes):
    # bytes를 Image 객체로 변환
    image = Image.open(io.BytesIO(image_bytes))

    # 이미지 표시
    plt.imshow(image)
    plt.axis('off')  # 축 제거
    plt.show()


def img_convert(image_bytes):
    return Image.open(io.BytesIO(image_bytes))


def show_bounding_box(result):
    # bounding box 정보

    boxes = result.boxes
    # print(type(boxes))
    # bbox의 위치 정보
    print(boxes.xyxy)  # bbox의 좌상단 우하단 x,y 좌표
    print(boxes.xyxyn)  # bbox의 좌상단 우하단 x,y 좌표 이미지크기 대비 비율

    print(boxes.xywh)  # bbox의 center x,y 좌표, bbox의 width, height 크기
    print(boxes.xywhn)  # bbox의 center x,y 좌표, bbox의 width, height 크기 이미지 크기 대비 비율

    # bbox 내의 object 분류
    print(boxes.cls)  # object class index
    print(boxes.conf)  # object들의 확률


def detection_to_json(boxes):
    # cls: class indices
    # conf: confidence scores
    # xyxy: bounding box coordinates (xmin, ymin, xmax, ymax)
    cls = boxes.cls.tolist() if isinstance(boxes.cls, torch.Tensor) else boxes.cls
    conf = boxes.conf.tolist() if isinstance(boxes.conf, torch.Tensor) else boxes.conf
    xyxy = boxes.xyxy.tolist() if isinstance(boxes.xyxy, torch.Tensor) else boxes.xyxy

    # Convert to list of detection dictionaries
    detections = []

    def dic_elements():
        dic = {
            "class": int(cls[i]),
            "confidence": float(conf[i]),
            "amount": float(predict_amount(bbox_size))
        }
        return dic

    def calc_bbox_size():
        return (xyxy[i][2] - xyxy[i][0]) * (xyxy[i][3] - xyxy[i][1])

    for i in range(len(cls)):
        bbox_size = calc_bbox_size()
        detections.append(dic_elements())

    # json 형식으로 변환
    json_result = json.dumps(detections, indent=4)
    return eval(json_result)


# 양추정 함수
def predict_amount(bbox_size):
    return 100


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
