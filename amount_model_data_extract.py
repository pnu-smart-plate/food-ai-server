import os
import csv
from PIL import Image
import torch
import ai

# 모델 불러오기
food_detect_ai = ai.DetectionModel()
tray_detect_ai = ai.TrayDetectionModel()

# 이미지 폴더 경로
image_folder = './images'
output_file = './output.csv'

# CSV 파일에 저장할 데이터 형식: 파일 이름, 비율
def save_to_csv(data, file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Food to Tray Ratio"])  # 헤더
        writer.writerows(data)

def img_convert(image_path):
    return Image.open(image_path)

def calc_bbox_size(xyxy, i):
    return (xyxy[i][2] - xyxy[i][0]) * (xyxy[i][3] - xyxy[i][1])

def detect_and_calculate_ratio(image_path):
    # 이미지 읽기
    with open(image_path, 'rb') as f:
        img_bytes = f.read()

    # 모델을 사용하여 예측
    food_predict = food_detect_ai.predict(img_convert(img_bytes))[0]
    tray_predict = tray_detect_ai.predict(img_convert(img_bytes))[0]

    food_bbox_size = calc_bbox_size(food_predict.boxes.xyxy.tolist(), 0)
    tray_bbox_size = calc_bbox_size(tray_predict.boxes.xyxy.tolist(), 0)

    # 비율 계산
    food_tray_ratio = food_bbox_size / tray_bbox_size
    return food_tray_ratio

# 폴더에서 이미지 파일들을 읽고 비율을 계산한 후 저장
def process_images(image_folder):
    results = []

    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = image_folder + '/' + filename
            print(image_path)
            ratio = detect_and_calculate_ratio(image_path)
            results.append([filename, ratio])

    return results

# 이미지 처리 및 결과 저장
if __name__ == "__main__":
    data = process_images(image_folder)
    save_to_csv(data, output_file)
    print(f'Results saved to {output_file}')
