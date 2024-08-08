from fastapi import FastAPI, UploadFile
import uvicorn
import ai

app = FastAPI()
detect_ai = ai.DetectionModel
amount_ai = ai.AmountModel


@app.post("/model/detect")
async def detect(file: UploadFile):
    img = await file.read()
    # 모델을 실행하고 결과를 받아오는 부분

    return file


@app.post("/model/amount")
async def amount():
    return {"message": "amount"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
