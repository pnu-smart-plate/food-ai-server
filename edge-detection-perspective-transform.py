from fileinput import filename

import numpy as np
import cv2
from scipy.spatial import distance as dist

from blacken_outside_bbox import blacken_outside_bbox


def wait():
    # 스페이스바가 눌려지기 전까지는 대기한다
    wait = cv2.waitKey(0)
    while (wait != 32):
        wait = cv2.waitKey(0)
        print(wait)


def mouse_handler(event, x, y, flags, param):  # 마우스로 좌표 알아내기
    if event == cv2.EVENT_LBUTTONUP:  # 왼쪽 버튼 누르면
        clicked = [x, y]  # 해당 좌표 저장
        print(clicked)


# 식판의 배경과 명함의 구분이 모호한 경우: 식판의 색깔과 테이블의 색깔이 유사할 때
# 이미지 전 처리로 grab_cut은 최대한 사용하지 않도록 보정하였다.
# 식품이 식판에 올라가있어서 grab_cut은 효과적이지 못하였다.
def grab_cut(resized):
    mask_img = np.zeros(resized.shape[:2], np.uint8)  # 초기 마스크를 만든다.

    # grabcut에 사용할 임시 배열을 만든다.
    bgdModel = np.zeros((1, 65), np.float64)  # background model array
    fgdModel = np.zeros((1, 65), np.float64)  # foreground model array

    # rect = (130, 51, 885-130, 661-51) #mouse_handler로 알아낸 좌표 / card1일때
    rect = (107, 89, 580, 467)  # card2 일 때 => 식판 위치로 다시 변경해야 됨

    ## 식판과 배경을 구분하는 grabCut
    cv2.grabCut(resized, mask_img, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)  # grabcut 실행

    mask_img = np.where((mask_img == 2) | (mask_img == 0), 0, 1).astype('uint8')  # 배경인 곳은 0, 그 외에는 1로 설정한 마스크를 만든다.
    img = resized * mask_img[:, :, np.newaxis]  # 이미지에 새로운 마스크를 곱해 배경을 제외한다.

    background = resized - img

    background[np.where((background >= [0, 0, 0]).all(axis=2))] = [0, 0, 0]

    img_grabcut = background + img

    cv2.imshow('grabcut', img_grabcut)
    wait()

    new_edged = edge_detection(img_grabcut)
    wait()

    global new_contour
    new_contour = contoursGrab(new_edged)


# 에지 검출 : 흑백 -> 가우시안블러링 -> 캐니
def edge_detection(img):
    # 그레이스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 완전한 검정색(0)과 그 외의 색상을 구분
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 윤곽선을 그리기 위한 빈 이미지 생성
    edge_img = np.zeros(img.shape[:2], dtype=np.uint8)

    # 찾은 윤곽선 그리기
    cv2.drawContours(edge_img, contours, -1, (255), 2)

    # 결과 표시
    cv2.imshow("Black color contours", edge_img)
    cv2.waitKey(0)

    return edge_img


def contours(edge,save_path):

    global checkpnt
    (cnts, _) = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for i in cnts:
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break
    else:
        if checkpnt == 0:
            checkpnt += 1
            grab_cut(resize_img)
            return
        else:
            print("No contour found")
            return

    # 투시 변환 적용
    warped = perspective_transform(resize_img, screenCnt)

    # edged = edge_detection(resize_img)
    (cnts, _) = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 계층관계가 필요없기 때문에 contour만 추출

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]  # contourArea : contour가 그린 면적

    for i in cnts:
        peri = cv2.arcLength(i, True)  # contour가 그리는 길이 반환
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # 길이에 2% 정도 오차를 둔다

        if len(approx) == 4:  # 도형을 근사해서 외곽의 꼭짓점이 4개라면 명암의 외곽으로 설정
            screenCnt = approx
            size = len(screenCnt)
            break
        if len(approx) != 4 and checkpnt == 0:  # 사각형이 그려지지 않는다면 grab_cut 실행
            size = 0
            checkpnt += 1
            grab_cut(resize_img)

        if len(approx) != 4 and checkpnt > 0:
            size = 0

    if (size > 0):
        # 2.A 원래 영상에 추출한 4 변을 각각 다른 색 선분으로 표시한다.
        cv2.line(resize_img, tuple(screenCnt[0][0]), tuple(screenCnt[size - 1][0]), (255, 0, 0), 3)
        for j in range(size - 1):
            color = list(np.random.random(size=3) * 255)
            cv2.line(resize_img, tuple(screenCnt[j][0]), tuple(screenCnt[j + 1][0]), color, 3)

        # for i in screenCnt: #이렇게 하면 네 변을 다른 색으로 표현 불가능(네 변이 모두 다 똑같은 색으로 나온다.)
        # color = list(np.random.random(size=3) * 256)
        # cv2.drawContours(resize_img, [screenCnt], -1,color, 3)

        # 2.B 추출된 선분(좌, 우, 상, 하)의 기울기, y절편, 양끝점의 좌표를 각각 출력
        axis = np.zeros(4)

        # 기울기 = (y증가량) / (x증가량)
        # left_axis = (screenCnt[0][0][1] - screenCnt[1][0][1]) / (screenCnt[0][0][0] - screenCnt[1][0][0])
        # down_axis = (screenCnt[1][0][1] - screenCnt[2][0][1]) / (screenCnt[1][0][0] - screenCnt[2][0][0])
        # right_axis = (screenCnt[2][0][1] - screenCnt[3][0][1]) / (screenCnt[2][0][0] - screenCnt[3][0][0])
        # upper_axis = (screenCnt[3][0][1] - screenCnt[0][0][1]) / (screenCnt[3][0][0] - screenCnt[0][0][0])

        axis[3] = (screenCnt[3][0][1] - screenCnt[0][0][1]) / (screenCnt[3][0][0] - screenCnt[0][0][0])
        for k in range(3):
            axis[k] = (screenCnt[k][0][1] - screenCnt[k + 1][0][1]) / (screenCnt[k][0][0] - screenCnt[k + 1][0][0])

        left_axis = axis[0]  # 좌 기울기
        down_axis = axis[1]  # 하 기울기
        right_axis = axis[2]  # 우 기울기
        upper_axis = axis[3]  # 상 기울기

        print("(2.B) 순서대로 좌, 우, 상, 하 선분의 기울기")
        print(left_axis, right_axis, upper_axis, down_axis)
        print("\n")

        # y = ax + b 에서 x = 0일때의 b가 y절편 / 기울기를 알고 두 좌표를 알 때의 방정식 : y - y1 = (y2 - y1)/(x2 - x1) * (x - x1)
        # 좌 선분의 y절편
        # left_y - screenCnt[1][0][1] = left_axis * (left_x - screenCnt[1][0][0])
        # left_y = (left_axis * left_x) - (left_axis * screenCnt[1][0][0]) + screenCnt[1][0][1]
        # 따라서 left_y = screenCnt[1][0][1] - (left_axis * screenCnt[1][0][0])
        left_y = screenCnt[1][0][1] - (left_axis * screenCnt[1][0][0])  # 좌 y절편

        # 우 선분의 y절편
        # right_y - screenCnt[3][0][1] = right_axis * (right_x - screenCnt[3][0][0])
        # right_y = (right_axis * right_x) - (right_axis * screenCnt[3][0][0]) + screenCnt[3][0][1]
        # 따라서 right_y = screenCnt[3][0][1] - (right_axis * screenCnt[3][0][0])
        right_y = screenCnt[3][0][1] - (right_axis * screenCnt[3][0][0])  # 우 y절편

        # 상 선분의 y절편
        # upper_y - screenCnt[0][0][1] = upper_axis * (upper_x - screenCnt[0][0][0])
        # upper_y = (upper_axis * upper_x) - (upper_axis * screenCnt[0][0][0]) + screenCnt[0][0][1]
        # 따라서 upper_y = screenCnt[0][0][1] - (upper_axis * screenCnt[0][0][0])
        upper_y = screenCnt[0][0][1] - (upper_axis * screenCnt[0][0][0])  # 상 y절편

        # 하 선분의 y절편
        # donw_y - screenCnt[2][0][1] = down_axis * (down_x - screenCnt[2][0][0])
        # down_y = (down_axis * down_x) - (down_axis * screenCnt[2][0][0]) + screenCnt[2][0][1]
        # 따라서 down_y = screenCnt[2][0][1] - (down_axis * screenCnt[2][0][0])
        down_y = screenCnt[2][0][1] - (down_axis * screenCnt[2][0][0])  # 하 y절편

        print("(2.B) 순서대로 좌, 우, 상, 하 선분의 y절편")
        print(left_y, right_y, upper_y, down_y)
        print("\n")

        # 양끝점의 좌표
        print("(2.B) 순서대로 좌, 우, 상, 하 선분의 양 끝점")
        print((screenCnt[0][0][0], screenCnt[0][0][1]), (screenCnt[1][0][0], screenCnt[1][0][1]))  # 좌 선분의 양 끝점
        print((screenCnt[2][0][0], screenCnt[2][0][1]), (screenCnt[3][0][0], screenCnt[3][0][1]))  # 우 성분의 양 끝점
        print((screenCnt[0][0][0], screenCnt[0][0][1]), (screenCnt[3][0][0], screenCnt[3][0][1]))  # 상 성분의 양 끝점
        print((screenCnt[1][0][0], screenCnt[1][0][1]), (screenCnt[2][0][0], screenCnt[2][0][1]))  # 하 성분의 양 끝점
        print("\n")

        # 3.B 네 꼭짓점을 각각 다른 색 점으로 표시한다.
        cv2.drawContours(resize_img, screenCnt, 0, (0, 0, 0), 15)  # 검
        cv2.drawContours(resize_img, screenCnt, 1, (255, 0, 0), 15)  # 파
        cv2.drawContours(resize_img, screenCnt, 2, (0, 255, 0), 15)  # 녹
        cv2.drawContours(resize_img, screenCnt, 3, (0, 0, 255), 15)  # 적

        cv2.imshow("With_Color_Image", resize_img)

        # 3.C  네 꼭지점(좌상, 좌하, 우상, 우하)의 좌표를 출력한다.
        vertex = solving_vertex(screenCnt.reshape(4, 2))
        # (topLeft, bottomLeft, topRight, bottomRight) = vertex

        print("(3.C) 순서대로 좌상, 좌하, 우상, 우하의 꼭짓점 좌표")
        print(vertex)

        # 결과 저장 및 표시
        cv2.imwrite(save_path, warped)
        print(f"Transformed image saved to {save_path}")
        cv2.imshow("Transformed Image", warped)
        cv2.waitKey(0)


def contoursGrab(edged):
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]  # contourArea : contour가 그린 면적

    largest = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    r = cv2.boxPoints(rect)
    box = np.int0(r)

    size = len(box)

    # 2.A 원래 영상에 추출한 4 변을 각각 다른 색 선분으로 표시한다.
    cv2.line(resize_img, tuple(box[0]), tuple(box[size - 1]), (255, 0, 0), 3)
    for j in range(size - 1):
        color = list(np.random.random(size=3) * 255)
        cv2.line(resize_img, tuple(box[j]), tuple(box[j + 1]), color, 3)

    # 4개의 점 다른색으로 표시
    boxes = [tuple(i) for i in box]
    cv2.line(resize_img, boxes[0], boxes[0], (0, 0, 0), 15)  # 검
    cv2.line(resize_img, boxes[1], boxes[1], (255, 0, 0), 15)  # 파
    cv2.line(resize_img, boxes[2], boxes[2], (0, 255, 0), 15)  # 녹
    cv2.line(resize_img, boxes[3], boxes[3], (0, 0, 255), 15)  # 적

    cv2.imshow("With_Color_Image", resize_img)

    return boxes


def order_dots(pts):
    p = np.array(pts)
    xSorted = p[np.argsort(p[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (a, c) = leftMost

    D = dist.cdist(a[np.newaxis], rightMost, "euclidean")[0]
    (b, d) = rightMost[np.argsort(D), :]
    return a, b, c, d


def transformationGrab(resized, pts):
    p = np.array(pts)
    rect = np.zeros((4, 2), dtype="float32")

    s = p.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    a, b, c, d = rect

    w1 = abs(c[0] - d[0])
    w2 = abs(a[0] - b[0])
    h1 = abs(b[1] - c[1])
    h2 = abs(a[1] - d[1])

    w = max([w1, w2])
    h = max([h1, h2])

    # dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.array([
        [0, 0],
        [640, 0],
        [640, 480],
        [0, 480]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    result = cv2.warpPerspective(resized, M, dsize=(640, 480))

    cv2.imshow("transformation", result)
    return result


def solving_vertex(pts):
    points = np.zeros((4, 2), dtype="uint32")  # x,y쌍이 4개 쌍이기 때문

    # 원점 (0,0)은 맨 왼쪽 상단에 있으므로, x+y의 값이 제일 작으면 좌상의 꼭짓점 / x+y의 값이 제일 크면 우하의 꼭짓점
    s = pts.sum(axis=1)
    points[0] = pts[np.argmin(s)]  # 좌상
    points[3] = pts[np.argmax(s)]  # 우하

    # 원점 (0,0)은 맨 왼쪽 상단에 있으므로, y-x의 값이 가장 작으면 우상의 꼭짓점 / y-x의 값이 가장 크면 좌하의 꼭짓점
    diff = np.diff(pts, axis=1)
    points[2] = pts[np.argmin(diff)]  # 우상
    points[1] = pts[np.argmax(diff)]  # 좌하

    src.append(points[0])
    src.append(points[1])
    src.append(points[2])
    src.append(points[3])

    return points


def transformation():
    # print(src)
    src_np = np.array(src, dtype=np.float32)
    # print(src_np)

    dst_np = np.array([
        [0, 0],
        [0, 480],
        [640, 0],
        [640, 480]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src=src_np, dst=dst_np)

    result = cv2.warpPerspective(resize_img, M=M, dsize=(640, 480))

    cv2.imshow("result", result)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"[{self.x} {self.y}]"


# y절편 구하기
def get_intercepts(ordered_dots):
    slopes = get_slopes(ordered_dots)

    a, b, c, d = [Point(*coord) for coord in ordered_dots]

    left = -a.x * slopes[0] + a.y
    right = -b.x * slopes[1] + b.y
    top = -a.x * slopes[2] + a.y
    bottom = -c.x * slopes[3] + c.y

    return left, right, top, bottom


# 기울기 구하기
def get_slopes(ordered_dots):
    a, b, c, d = [Point(*coord) for coord in ordered_dots]

    left = (a.y - c.y) / (a.x - c.x)
    right = (b.y - d.y) / (b.x - d.x)
    top = (a.y - b.y) / (a.x - b.x)
    bottom = (c.y - d.y) / (c.x - d.x)

    return left, right, top, bottom


def perspective_transform(image, screenCnt):
    # 꼭짓점 순서 정렬 (좌상, 우상, 우하, 좌하)
    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # 변환 후 이미지 크기 설정
    (maxWidth, maxHeight) = (640, 480)

    # 변환 후 꼭짓점 좌표
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # 투시 변환 행렬 계산
    M = cv2.getPerspectiveTransform(rect, dst)

    # 투시 변환 적용
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped






if __name__ == "__main__":
    filename = 'soup2.jpeg'
    original_image_path = "./images/"+filename

    # TODO: predict bbox의 xywh값 자동 입력 구현하기
    xywh = [280.8320, 223.6365, 541.6607, 388.0640]  # [center_x, center_y, width, height]
    output_path = "./images/"+filename[:-5]+"preprocess.jpeg"

    # 바운딩 박스 근처로 이미지 전처리
    blacken_outside_bbox(original_image_path, xywh, margin_ratio=0.08, output_path=output_path)

    preprocessed_image_path = output_path
    ori_img = cv2.imread(preprocessed_image_path)
    resize_img = cv2.resize(ori_img, dsize=(640, 480), interpolation=cv2.INTER_AREA)  # 원래 사진이 4000x3000 이라서 사이즈 조절하였음.

    global checkpnt
    global new_contour
    checkpnt = 0
    src = []  # 식판의 꼭짓점
    cv2.imshow('original_img', resize_img)
    wait()

    # 1. 그레이 스케일 변환
    # 2. 이진화
    # 3. 윤곽선 찾기
    # 4. 윤곽선 그리기(검정색 배경의 빈 이미지에 윤곽선만 그림)
    edged = edge_detection(resize_img)
    wait()

    # 1. edge 이미지에서 윤곽선 찾기
    # 2. 윤곽선 정렬(면적 기준 상위 5개의 윤곽선만 선택)
    # 3. 사각형 찾기
    # 4-1. 사각형 없을 경우 grab_cut 함수를 호출하여 이미지 처리
    # 4-2. 사각형이 있을 경우, 사각형 윤곽선 그리고 각 변의 기울기, y 절편, 꼭짓점 좌표 정보 계산
    # 5. 이미지명+result.jpeg로 결과 이미지 파일 저장
    contours(edged,preprocessed_image_path[:-5]+"result.jpeg")
    wait()
    if checkpnt == 0:
        transformation()
    else:
        pts = new_contour
        transformationGrab(resize_img, pts)
        dots = order_dots(pts)
        a, b, c, d = dots
        print("2-B	추출된 네변(선분), (즉, 좌, 우, 상, 하단 )의 기울기, y 절편, 양끝점의 좌표을 각각 출력할 것.\n")
        print("기울기 :")
        print(get_slopes(dots))
        print("\n")
        print("y절편 :")
        print(get_intercepts(dots))
        print("\n")
        print("양끝점 :")
        print(a, c)
        print(b, d)
        print(a, b)
        print(c, d)

        print("\n")
        print("3-C.	네 꼭지점(좌상, 좌하, 우상, 우하 코너)의 좌표를 출력한다")
        print(a)
        print(b)
        print(c)
        print(d)


