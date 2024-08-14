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
    # img_show(img)
    # 모델을 실행하고 결과를 받아오는 부분
    predict = detect_ai.predict(img_convert(img))
    print(predict)
    return file


# @app.post("/model/amount")
async def amount():
    return {"message": "amount"}


def img_show(image_bytes):
    # bytes를 Image 객체로 변환
    image = Image.open(io.BytesIO(image_bytes))

    # 이미지 표시
    plt.imshow(image)
    plt.axis('off')  # 축 제거
    plt.show()


def img_convert(image_bytes):
    return Image.open(io.BytesIO(image_bytes))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
