""" 
Make yolo v7 ECG Box data 

input -> image
process -> drag & drop
output -> boxed image & box coordinates

coordinate type ( box_mid_x, box_mid_y, width, height ) - normalized value

"""
import random
import shutil

import cv2
import os
import numpy as np

from scipy.stats import norm
import json

from tqdm import tqdm

# 전역 변수 설정
rectangles = []  # 사각형 좌표를 저장할 리스트
start_point = None
end_point = None
drawing = False
delete_mode = False  # 삭제 모드 플래그
image = None
img_copy = None
img_with_text = None  # 텍스트가 추가된 이미지


# 배경 추가 ##############################################################################################################
def add_background_noise(img, background_img, boxes):

   
    # 패딩 값 설정 (상단, 하단, 좌측, 우측)
    top, bottom, left, right = 800, 800, 800, 800  

    
    """
    Add Gaussian noise to the background outside the bounding boxes.
    """
    original_img = img.copy()
    # 패딩 추가하기
    extended_img = cv2.copyMakeBorder(
        original_img, top, bottom, left, right, 
        borderType=cv2.BORDER_CONSTANT, 
        value=[0,0,0]  # 패딩 색상: 흰색
    )

    '''확률적으로 이미지가 회전되는 augmentation 넣기'''
   
    background_img = cv2.resize(background_img, (extended_img.shape[1], extended_img.shape[0]))

    x_start = 0  # 좌측 패딩
    x_end = background_img.shape[1]  # 우측 패딩 끝
    y_start = 0  # 상단 패딩
    y_end = background_img.shape[0]  # 하단 패딩 끝

    # extended_img[0:top,:,:] = background_img[0:top,:,:] 
    # extended_img[:,0:left,:] = background_img[:,0:left,:] 
    # extended_img[:,-right:,:] = background_img[:,-right:,:] 
    # extended_img[-bottom:,:,:] = background_img[-bottom:,:,:]
    
    mask = np.all(extended_img == 0, axis=-1)  # 모든 채널이 0인 부분 찾기 (3채널 기준)
    extended_img[mask] = background_img[mask]  # 마스크된 부분을 overlay_img 픽셀로 교체
     

    

    # h, w = img.shape[:2]
    # mask = np.ones((h, w), dtype=np.uint8)  # Mask to exclude bounding boxes

    # # Mark bounding boxes in the mask
    # for start, end in boxes:
    #     x1, y1 = start
    #     x2, y2 = end
    #     mask[y1:y2, x1:x2] = 0  # Zero means inside box, 1 means background

    # # Add Gaussian noise only where the mask is 1 (background)
    # noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    # noisy_img[mask == 1] = cv2.add(noisy_img, noise)[mask == 1]

    return extended_img

def clip_bounding_box(x1, y1, x2, y2, img_shape):
    """
    Clip bounding boxes to ensure they are within the image boundaries.
    """
    h, w = img_shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return (x1, y1), (x2, y2)

def warp_image(image: np.ndarray, amplitude_x: float = 5, amplitude_y: float = 5):
    # 원본 이미지 크기 가져오기
    height, width = image.shape[:2]

    # 사인파 주기 설정
    frequency_x = 1 / (width / 2)  # 수평 방향 주기를 이미지 폭의 한 파장으로 설정
    frequency_y = 1 / height  # 수직 방향 주기를 이미지 높이의 반파장으로 설정

    # 여유 공간을 추가하여 잘리는 부분 방지
    new_height = int(height + 2 * amplitude_y)
    new_width = int(width + 2 * amplitude_x)
    warped_image = np.zeros(
        (new_height, new_width, *image.shape[2:]), dtype=image.dtype
    )

    # 이미지 중심 위치 계산
    y_offset = amplitude_y
    x_offset = amplitude_x

    # 각 픽셀 위치를 사인 함수에 따라 이동
    for y in range(height):
        for x in range(width):
            # 사인 함수를 사용하여 수평 및 수직 이동량 계산
            offset_x = int(
                amplitude_x * np.sin(2 * np.pi * frequency_y * y)
            )  # 높이 기준 반파장
            offset_y = int(
                amplitude_y * np.sin(2 * np.pi * frequency_x * x)
            )  # 폭 기준 한 파장

            # 새로운 좌표에 이미지 픽셀 배치 (여유 공간을 포함한 좌표)
            new_x = x + x_offset + offset_x
            new_y = y + y_offset + offset_y

            # 좌표가 유효한 범위 내에 있을 때만 픽셀 배치
            if 0 <= new_x < new_width and 0 <= new_y < new_height:
                warped_image[new_y, new_x] = image[y, x]

    return warped_image



def get_warp(image_list):
    warped_list = []
    for image in image_list:
        warped_list.append(warp_image(image))
    return warped_list


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
    image = add_background_noise(image)
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

def process_json_files(input_folder):
    # 폴더 내 모든 json 파일 읽기
    for root, dirs, files in os.walk(input_folder):
        for file_name in tqdm(files, total=len(files), desc=f"converting json to txt"):
            if file_name.endswith(".json"):
                json_path = os.path.join(root, file_name)

                # json 파일 열기
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)


                    # json 파일 이름에 txt를 붙여서 저장할 파일 경로 생성
                    txt_file_name = os.path.splitext(file_name)[0] + ".txt"
                    txt_file_path = os.path.join(root, txt_file_name)

                    width_from_json = data.get("width")
                    height_from_json = data.get("height")


                    # 파일 쓰기 모드로 텍스트 파일 열기
                    with open(txt_file_path, "w") as txt_file:
                        # "leads"에서 "lead_name"과 "lead_bounding_box"를 읽음
                        leads = data.get("leads", [])

                        # 13행의 x, y, w, h 값 쓰기 (lead_bounding_box로부터 계산)
                        for lead in leads:
                            lead_name = lead.get("lead_name")
                            bounding_box = lead.get("lead_bounding_box", {})


                            # 좌표 순서: 1(좌하단), 2(우상단), 3(좌상단), 4(우하단)
                            if len(bounding_box) == 4:
                                y1, x1 = bounding_box.get("0", [0, 0])  # 좌하단
                                y2, x2 = bounding_box.get("1", [0, 0])  # 우하단
                                y3, x3 = bounding_box.get("2", [0, 0])  # 우상단
                                y4, x4 = bounding_box.get("3", [0, 0])  # 좌상단

                                # x, y, w, h 계산
                                w_pad = (x3 - x1) * 0.1
                                h_pad = (y3 - y1) * 0.2

                                x = ((x1 + x3) / 2) / width_from_json  # x좌표의 센터
                                y = ((y2 + y4) / 2) / height_from_json  # y좌표의 센터
                                w = (x3 - x1 + w_pad) / width_from_json  # 너비 (우상단 x - 좌하단 x)
                                if w > 1:
                                    w = 1


                                h = (y3 - y1 + h_pad) / height_from_json  # 높이 (우상단 y - 좌하단 y)

                                # txt 파일에 기록할 13 x 4 행렬 형식의 행을 추가 (실수로 기록)

                                if  w > 0.6:
                                    w -= 0.5 * (w_pad / width_from_json)
                                    row = f"2 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                                    txt_file.write(row)
                                else:
                                    row = f"1 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                                    txt_file.write(row)
                except:
                    pass

root = "/media/jwlee/9611c7a0-8b37-472c-8bbb-66cac63bc1c7/image_dataset/Generated_Images"

def main():
    input_folder = root
    folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
    for per, folder in enumerate(folders):
        print(f"processed : {per}/{len(folders)}")
        process_json_files(os.path.join(input_folder,folder))

    for per, folder in enumerate(folders):

        directory = os.path.join(input_folder, folder)
        output_directory = os.path.join(input_folder, "merged")
        print("current : ",directory, " processed : ", per ,"/",len(folders))


        file_list = []

        sub_folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

        for sub_folder in sub_folders:
            current_dir = os.path.join(directory, sub_folder)
            image_files = [
                f for f in os.listdir(current_dir)
                if (f.endswith((".jpg", ".png", ".jpeg")) and "segmentation" not in f)
            ]  # 이미지 파일

            segmentation_files = [f for f in os.listdir(current_dir)
                if (f.endswith((".jpg", ".png", ".jpeg")) and "segmentation" in f)]
            label_files = [
                f for f in os.listdir(current_dir) if f.endswith((".txt"))
            ]  # 라벨 파일
            hea_files = [
                f for f in os.listdir(current_dir) if f.endswith((".hea"))
            ]
            json_files = [
                f for f in os.listdir(current_dir) if f.endswith((".json"))
            ]
            if not image_files:
                continue

            random.seed(42)
            image_files = random.sample(image_files,5)
            # label_files = random.sample(label_files,10)
            # hea_files = random.sample(hea_files,10)
            # json_files = random.sample(json_files,10)


            for file in tqdm(image_files, total=len(image_files), desc="sample datas"):
                # 대상 디렉토리 생성
                target_dir = os.path.join("/media/jwlee/9611c7a0-8b37-472c-8bbb-66cac63bc1c7/ECG_yolov7_datasets_ver2")

                os.makedirs(target_dir, exist_ok=True)
                os.makedirs(os.path.join(target_dir, "images"), exist_ok=True)
                os.makedirs(os.path.join(target_dir, "labels"), exist_ok=True)
                try:
                    shutil.copy(
                        src=os.path.join(current_dir, file),

                        dst=os.path.join(target_dir, "images", file)
                    )
                    # shutil.copy(
                    #     src=os.path.join(current_dir, file[:-4] + "_segmentation" + file[-4:]),
                    #     dst=os.path.join(target_dir, file[:-4] + "_segmentation" + file[-4:])
                    # )


                    shutil.copy(
                        src=os.path.join(current_dir, file[:-4] + ".txt"),
                        dst=os.path.join(target_dir, "labels", file[:-4] + ".txt")
                    )

                    # shutil.copy(
                    #     src=os.path.join(current_dir, file[:-6] + ".hea"),
                    #     dst=os.path.join(target_dir, file[:-4] + ".hea")
                    # )
                    # shutil.copy(
                    #     src=os.path.join(current_dir, file[:-4] + ".json"),
                    #     dst=os.path.join(target_dir, file[:-4] + ".json")
                    # )
                except:
                    pass


            file_idx = 0
            while file_idx < len(image_files):
                file_name = image_files[file_idx]
                image_path = f"{current_dir}/{file_name}"
                txt_path = f"{current_dir}/{file_name[:-4]}.txt"
                modified_txt_path = f"{current_dir}/modified/{file_name[:-4]}.txt"
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




if __name__ == "__main__":
    bg_path = os.path.expanduser(os.path.join(os.getcwd(), "bg.png"))
    bg = cv2.imread(bg_path)
    # import pdb; pdb.set_trace()
    input_folder = root
    folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
    for per, folder in enumerate(folders):

        directory = os.path.join(input_folder, folder)
        output_directory = os.path.join(input_folder, "merged")
        print("current : ",directory, " processed : ", per ,"/",len(folders))


        file_list = []

        sub_folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

        for sub_folder in sub_folders:
            current_dir = os.path.join(directory, sub_folder)
            image_files = [
                f for f in os.listdir(current_dir)
                if (f.endswith((".jpg", ".png", ".jpeg")) and "segmentation" not in f)
            ]  # 이미지 파일


            for file in image_files:
                # Load Image
                image_path = os.path.join(current_dir, file)
                image = cv2.imread(image_path)

                # Ensure image is loaded
                if image is None:
                    print(f"Error loading image: {image_path}")
                    

                # Add Noise Outside Bounding Boxes
                _, boxes = load_yolo_data(os.path.join(current_dir, file[:-4] + ".txt"), image.shape)
                clipped_boxes = [clip_bounding_box(x1, y1, x2, y2, image.shape) for (x1, y1), (x2, y2) in boxes]

                noisy_image = add_background_noise(image, bg, clipped_boxes)

                # Save Processed Image
                # output_image_path = os.path.join(target_dir, "images", file)
                # cv2.imwrite(output_image_path, noisy_image)
                cv2.namedWindow("Image", flags=cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Image", width=1700, height=900)
                cv2.imshow('Image',noisy_image)
                cv2.waitKey(0)
                cv2.distroyAllWindows()
                # print(f"Processed image saved to: {output_image_path}")

    # main()




