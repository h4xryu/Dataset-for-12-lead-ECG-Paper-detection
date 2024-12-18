import random
import shutil


import os
import numpy as np

from scipy.stats import norm
import json

from tqdm import tqdm

import cv2

# YOLO 형식 좌표 불러오기 함수
def load_yolo_data(file_path, img_shape):
    h, w = img_shape[:2]  # 실제 이미지 크기
    boxes = []
    classes = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split()
                if len(data) != 5:
                    continue
                class_id = data[0]  # 클래스 ID
                x_center, y_center, box_width, box_height = map(float, data[1:])

                # YOLO 좌표는 정규화된 값이므로 이미지 크기에 맞게 변환
                x1 = int((x_center - box_width / 2) * w)
                y1 = int((y_center - box_height / 2) * h)
                x2 = int((x_center + box_width / 2) * w)
                y2 = int((y_center + box_height / 2) * h)

                boxes.append(((x1, y1), (x2, y2)))  # 좌상단과 우하단 좌표로 변환
                classes.append(class_id)
    return classes, boxes


# YOLO 형식으로 좌표 저장 함수
def save_yolo_data(rectangles, img_shape, output_txt):
    h, w = img_shape[:2]
    with open(output_txt, "w") as f:
        for start, end in rectangles:
            x1, y1 = start
            x2, y2 = end

            # 좌표 정렬
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])

            # YOLO 형식 좌표 계산
            box_mid_x = ((x_min + x_max) / 2) / w
            box_mid_y = ((y_min + y_max) / 2) / h
            box_width = (x_max - x_min) / w
            box_height = (y_max - y_min) / h

            # 파일에 저장 (클래스 ID는 1로 설정)
            f.write(
                f"1 {box_mid_x:.6f} {box_mid_y:.6f} {box_width:.6f} {box_height:.6f}\n"
            )


# YOLO 좌표를 텍스트로 변환하는 함수
def yolo_to_text(rectangles, img_shape, classes):
    h, w = img_shape[:2]
    text_lines = []
    for i, (start, end) in enumerate(rectangles):
        x1, y1 = start
        x2, y2 = end

        # 좌표 정렬
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        # YOLO 형식 좌표 계산
        box_mid_x = ((x_min + x_max) / 2) / w
        box_mid_y = ((y_min + y_max) / 2) / h
        box_width = (x_max - x_min) / w
        box_height = (y_max - y_min) / h

        # 텍스트 라인 생성
        text_lines.append(
            f"{classes[0]} {box_mid_x:.6f} {box_mid_y:.6f} {box_width:.6f} {box_height:.6f}"
        )
    return text_lines


# 텍스트를 이미지에 그리는 함수
def draw_text_on_image(img_with_text, text_lines):
    y0, dy = 30, 25  # 텍스트 시작 위치와 줄 간격
    for i, line in enumerate(text_lines):
        y = y0 + i * dy
        cv2.putText(
            img_with_text,
            line,
            (img_with_text.shape[1] - 400, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


# 이미지에 텍스트 영역을 추가하는 함수
def add_text_area_to_image(img):
    h, w, _ = img.shape
    new_w = w + 400  # 오른쪽에 400px 공간 추가
    img_with_text = np.zeros((h, new_w, 3), dtype=np.uint8)
    img_with_text[:, :w] = img  # 기존 이미지를 복사
    return img_with_text


# 마우스 이벤트 콜백 함수
def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, img_copy, img_with_text, rectangles, delete_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        if delete_mode:
            # 클릭한 좌표가 어떤 사각형 내부에 있는지 확인
            for i, (start, end) in enumerate(rectangles):
                x1, y1 = start
                x2, y2 = end
                x_min, x_max = sorted([x1, x2])
                y_min, y_max = sorted([y1, y2])
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    # 사각형 삭제
                    rectangles.pop(i)
                    # 이미지 업데이트
                    img_copy = image.copy()
                    img_with_text = add_text_area_to_image(img_copy)  # 텍스트 영역 추가
                    for rect in rectangles:
                        cv2.rectangle(img_with_text, rect[0], rect[1], (0, 200, 0), 2)
                    text_lines = yolo_to_text(rectangles, image.shape)
                    draw_text_on_image(img_with_text, text_lines)  # 텍스트 갱신
                    cv2.imshow("Image", img_with_text)
                    break  # 한 개의 사각형만 삭제
        else:
            # 새로운 사각형 그리기 시작
            drawing = True
            start_point = (x, y)
            end_point = start_point

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            img_copy = image.copy()
            img_with_text = add_text_area_to_image(img_copy)  # 텍스트 영역 추가
            # 기존 사각형들 그리기
            for rect in rectangles:
                cv2.rectangle(img_with_text, rect[0], rect[1], (0, 200, 0), 2)
            # 현재 그리는 사각형 그리기
            cv2.rectangle(img_with_text, start_point, end_point, (0, 0, 200), 2)
            cv2.imshow("Image", img_with_text)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            end_point = (x, y)
            rectangles.append((start_point, end_point))
            img_copy = image.copy()
            img_with_text = add_text_area_to_image(img_copy)  # 텍스트 영역 추가
            # 모든 사각형 다시 그리기
            for rect in rectangles:
                cv2.rectangle(img_with_text, rect[0], rect[1], (0, 200, 0), 2)
            # YOLO 형식의 텍스트 변환 및 이미지에 출력
            text_lines = yolo_to_text(rectangles, image.shape)
            draw_text_on_image(img_with_text, text_lines)
            cv2.imshow("Image", img_with_text)


# 이미지 로드 및 박스 불러오기
def load_image_and_boxes(img_path, yolo_file):
    global image, img_copy, img_with_text, rectangles
    image = cv2.imread(img_path)
    if image is None:
        print("Error: 이미지를 로드할 수 없습니다.")
        return
    img_copy = image.copy()
    img_with_text = add_text_area_to_image(img_copy)  # 텍스트 영역 추가
    classes, rectangles = load_yolo_data(yolo_file, image.shape)
    # 모든 사각형 그리기
    for i, rect in enumerate(rectangles):
        cv2.putText(
            img=img_with_text,
            text=classes[i],
            org=(rect[0][0], rect[0][1]),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 0),
            thickness=3,
        )
        cv2.rectangle(img_with_text, rect[0], rect[1], (0, 200, 0), 2)
    # YOLO 형식의 텍스트 변환 및 이미지에 출력
    text_lines = yolo_to_text(rectangles, image.shape, classes)
    draw_text_on_image(img_with_text, text_lines)


# 이미지 표시 및 키 이벤트 처리
def display_image_with_boxes(output_txt):
    global img_with_text, delete_mode
    while True:
        # img_with_text = cv2.resize(img_with_text, (960, 540))
        cv2.namedWindow("Image", flags=cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", width=1700, height=900)
        cv2.imshow("Image", img_with_text)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return 1
        elif key == ord("e"):
            return -1
        elif key == ord("s"):
            save_yolo_data(rectangles, image.shape, output_txt)
            print(f"좌표가 {output_txt} 파일에 저장되었습니다.")
            break
        elif key == ord("d"):
            delete_mode = not delete_mode
            if delete_mode:
                print("삭제 모드 활성화")
            else:
                print("삭제 모드 비활성화")
        elif key == ord("p"):
            print("문제가 생긴 포인트")
            break





root = "/media/jwlee/9611c7a0-8b37-472c-8bbb-66cac63bc1c7/ECG_yolov7_datasets_augmented"

def main():
    input_folder = root
    folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
    for per, folder in enumerate(folders):
        print(f"folder : {per}/{len(folders)}")

    for per, folder in enumerate(folders):

        directory = os.path.join(input_folder, folder)
        output_directory = os.path.join(input_folder, "merged")
        print("current : ",directory)


        file_list = []

        sub_folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

        images_dir = os.path.join(directory, "images")
        labels_dir = os.path.join(directory, "labels")

        image_files = [
            f for f in os.listdir(images_dir) if (f.endswith((".jpg", ".png", ".jpeg")) and "segmentation" not in f)
        ]  # 이미지 파일


        file_idx = 0
        while file_idx < len(image_files):
            file_name = image_files[file_idx]
            image_path = f"{images_dir}/{file_name}"
            txt_path = f"{labels_dir}/{file_name[:-4]}.txt"
            # modified_txt_path = f"{current_dir}/modified/{file_name[:-4]}.txt"
            print("처리중 파일:")
            print(image_path)
            print(txt_path)
            load_image_and_boxes(
                image_path,
                txt_path
            )
            cv2.namedWindow("Image", flags=cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image", width=640, height=640)
            prev_flag = display_image_with_boxes(output_txt=txt_path)
            cv2.destroyAllWindows()
                        
            if prev_flag == -1:
                if file_idx > 0:
                    file_idx -= 1
                    continue
                else :
                    print("첫 번째 인덱스 입니다.")
            elif prev_flag == 1:
                file_idx += 1
                continue
            

main()