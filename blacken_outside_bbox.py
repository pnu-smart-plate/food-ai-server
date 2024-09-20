import cv2
import numpy as np


def blacken_outside_bbox(image_path, xywh, margin_ratio=0.08, output_path=None):
    """
    이미지에서 지정된 바운딩 박스 외부와 안쪽 일부를 검정색으로 채색.

    :param image_path: 입력 이미지 파일 경로
    :param xywh: 바운딩 박스 좌표 [center_x, center_y, width, height]
    :param margin_ratio: 바운딩 박스 크기에 대한 마진 비율 (기본값: 0.08, 즉 8%)
    :param output_path: 출력 이미지 저장 경로 (기본값: None, 지정하지 않으면 화면에 표시만 함)
    """
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:  # 잘못된 경로
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

    # xywh를 [x, y, w, h] 형식으로 변환
    center_x, center_y, width, height = xywh
    x = int(center_x - width / 2)
    y = int(center_y - height / 2)
    w = int(width)
    h = int(height)

    # 마진 계산 (바운딩 박스 크기의 margin_ratio*100%(기본값:1%)로 설정)
    margin_w = int(w * margin_ratio)
    margin_h = int(h * margin_ratio)

    # 바운딩 박스 좌표 조정
    x += margin_w
    y += margin_h
    w -= 2 * margin_w
    h -= 2 * margin_h

    # 이미지 크기 내에서 바운딩 박스 좌표 조정
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(img.shape[1] - x, w))  # 최소 너비를 1로 설정
    h = max(1, min(img.shape[0] - y, h))  # 최소 높이를 1로 설정

    # 마스크 생성
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[y:y + h, x:x + w] = 255

    # 마스크를 이용하여 바운딩 박스 외부를 검정색으로 채색
    result = img.copy()
    result[mask == 0] = [0, 0, 0]

    # 결과 저장 또는 표시
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"처리된 이미지가 저장되었습니다: {output_path}")
    else:
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


if __name__ == "__main__":
    image_path = "soup2.jpeg"  # 이미지 경로

    # TODO: predict bbox의 xywh값 자동 입력 구현하기
    xywh = [280.8320, 223.6365, 541.6607, 388.0640]  # [center_x, center_y, width, height]
    output_path = "./testimageresult.jpeg"  # 이미지 저장 결과

    blacken_outside_bbox(image_path, xywh, margin_ratio=0.08, output_path=output_path)
