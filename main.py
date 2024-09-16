import json
import torch
from fastapi import FastAPI, UploadFile
import uvicorn
import ai

import io
from PIL import Image
import matplotlib.pyplot as plt

app = FastAPI()
food_detect_ai = ai.DetectionModel()
tray_detect_ai = ai.TrayDetectionModel()
amount_ai = ai.AmountModel()


@app.post("/analyze")
async def detect(file: UploadFile):
    img = await file.read()

    # 모델을 실행하고 결과를 받아오는 부분
    food_predict = food_detect_ai.predict(img_convert(img))[0]
    tray_predict = tray_detect_ai.predict(img_convert(img))[0]

    # get_bounding_box(x)
    result_json = detection_to_json(food_predict.boxes, tray_predict.boxes)
    # img_show(img)
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


def detection_to_json(food_boxes, tray_boxes):
    # cls: class indices
    # conf: confidence scores
    # xyxy: bounding box coordinates (xmin, ymin, xmax, ymax)
    food_cls = food_boxes.cls.tolist() if isinstance(food_boxes.cls, torch.Tensor) else food_boxes.cls
    food_conf = food_boxes.conf.tolist() if isinstance(food_boxes.conf, torch.Tensor) else food_boxes.conf
    food_xyxy = food_boxes.xyxy.tolist() if isinstance(food_boxes.xyxy, torch.Tensor) else food_boxes.xyxy

    # tray_cls = tray_boxes.cls.tolist() if isinstance(tray_boxes.cls, torch.Tensor) else tray_boxes.cls
    # tray_conf = tray_boxes.conf.tolist() if isinstance(tray_boxes.conf, torch.Tensor) else tray_boxes.conf
    tray_xyxy = tray_boxes.xyxy.tolist() if isinstance(tray_boxes.xyxy, torch.Tensor) else tray_boxes.xyxy

    # Convert to list of detection dictionaries
    detections = []

    def dic_elements():
        dic = {
            "class": int(food_cls[i]),
            "confidence": float(food_conf[i]),
            "amount": float(predict_amount(food_cls[i], food_bbox_size, tray_bbox_size))
        }
        return dic

    def calc_bbox_size(xyxy, i):
        return (xyxy[i][2] - xyxy[i][0]) * (xyxy[i][3] - xyxy[i][1])

    # 최종 결과 json 만들기
    tray_bbox_size = calc_bbox_size(tray_xyxy, 0)
    for i in range(len(food_cls)):
        food_bbox_size = calc_bbox_size(food_xyxy, i)
        detections.append(dic_elements())

    # json 형식으로 변환
    json_result = json.dumps(detections, indent=4)
    return eval(json_result)


# 양추정 함수
def predict_amount(cls, food_bbox_size, tray_bbox_size):
    print(f"Class: {cls}, Tray bbox size: {tray_bbox_size}, Food bbox size: {food_bbox_size}, Food bbox / tray bbox: {food_bbox_size / tray_bbox_size}")
    # todo 양추정 모델 완성해서 결과 가져오기
    return 100


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
