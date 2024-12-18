import cv2
import os
import numpy as np
import random
from tqdm import tqdm
from scipy.stats import norm
import shutil
from sklearn.model_selection import train_test_split
from skimage.transform import warp

# 전역 변수 설정
rectangles = []  # 사각형 좌표를 저장할 리스트
start_point = None
end_point = None
drawing = False
delete_mode = False  # 삭제 모드 플래그
image = None
img_copy = None
img_with_text = None  # 텍스트가 추가된 이미지

# 배경 추가 함수
def add_background_noise(img, background_img, label_file):
    rval = random.randint(100, 1000)
    top = bottom = left = right = rval

    original_img = img.copy()
    h, w = original_img.shape[:2]

    angle = random.randint(-5, 5)
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    original_img = cv2.warpAffine(original_img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

    extended_img = cv2.copyMakeBorder(
        original_img, top, bottom, left, right, 
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    hh, ww = extended_img.shape[:2]
    boxes = []
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split()
                if len(data) != 5:
                    continue
                class_id, x_center, y_center, box_width, box_height = map(float, data)

                x1 = int((x_center - box_width / 2) * w)
                y1 = int((y_center - box_height / 2) * h)
                x2 = int((x_center + box_width / 2) * w)
                y2 = int((y_center + box_height / 2) * h)

                x1 += left
                y1 += top
                x2 += left
                y2 += top

                x_center = ((x1 + x2) / 2) / ww
                y_center = ((y1 + y2) / 2) / hh
                box_width = (x2 - x1) / ww
                box_height = (y2 - y1 + (abs(angle)*10) * ((y2 - y1) * 0.2)) / hh

                if len(boxes) < 13:
                    boxes.append([class_id, x_center, y_center, box_width, box_height])

    background_img = cv2.resize(background_img, (extended_img.shape[1], extended_img.shape[0]))
    mask = np.all(extended_img == 0, axis=-1)
    extended_img[mask] = background_img[mask]

    return extended_img, boxes

# 그림자 효과 생성 함수
def generate_spot_light_mask(mask_size, position=None, max_brightness=255, min_brightness=0, mode="gaussian", linear_decay_rate=None):
    if position is None:
        position = [(random.randint(0, mask_size[0]), random.randint(0, mask_size[1]))]
    if mode == "gaussian":
        mu = np.sqrt(mask_size[0] ** 2 + mask_size[1] ** 2)
        dev = mu / 3.5
        mask = _decay_value_radically_norm_in_matrix(mask_size, position, max_brightness, min_brightness, dev)
    mask = np.asarray(mask, dtype=np.uint8)
    mask = cv2.medianBlur(mask, 5)
    mask = 255 - mask
    return mask

def _decay_value_radically_norm_in_matrix(mask_size, centers, max_value, min_value, dev):
    center_prob = norm.pdf(0, 0, dev)
    x_value_rate = np.zeros((mask_size[1], mask_size[0]))
    for center in centers:
        coord_x = np.arange(mask_size[0])
        coord_y = np.arange(mask_size[1])
        xv, yv = np.meshgrid(coord_x, coord_y)
        dist_x = xv - center[0]
        dist_y = yv - center[1]
        dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
        x_value_rate += norm.pdf(dist, 0, dev) / center_prob
    mask = x_value_rate * (max_value - min_value) + min_value
    mask[mask > 255] = 255
    return mask

def add_spot_light(image, light_position=None, max_brightness=255, min_brightness=0, mode="gaussian", transparency=None):
    if transparency is None:
        transparency = random.uniform(0.5, 0.85)
    frame = np.copy(image)
    height, width, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = generate_spot_light_mask((width, height), light_position, max_brightness, min_brightness, mode)
    hsv[:, :, 2] = hsv[:, :, 2] * transparency + mask * (1 - transparency)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    frame[frame > 255] = 255
    return np.asarray(frame, dtype=np.uint8)

# 왜곡 효과 생성 함수
def add_bend_distortion(original):
    height, width = original.shape[:2]
    horizontal_bend = random.choice([0, 0.005, -0.005, 0.006, -0.006, 0.008, -0.008])
    vertical_bend = random.choice([0, 0.005, -0.005, 0.006, -0.006, 0.008, -0.008])

    def transform_coords(coords):
        x, y = coords[:, 0], coords[:, 1]
        y_new = y + horizontal_bend * width * np.sin(2 * np.pi * x / width)
        x_new = x + vertical_bend * width * np.sin(2 * np.pi * x / width)
        return np.vstack([x_new, y_new]).T

    return warp(original, transform_coords, mode="constant", cval=0, preserve_range=True).astype(original.dtype)

root = "/media/jwlee/9611c7a0-8b37-472c-8bbb-66cac63bc1c7/"
version = "ECG_yolov7_datasets"
set_list = ["train", "test", "valid"]
folder_list = ["images", "labels"]
bg_path = os.path.expanduser(os.path.join(os.getcwd(), "bg_noises"))
root += version

if __name__ == "__main__":
    for set in set_list:
        print(f"Current set: {set}")
        folder = folder_list[0]
        current_path = os.path.join(root, set, folder)
        os.makedirs(current_path, exist_ok=True)
        print(current_path)

        if os.path.exists(current_path):
            image_directory = os.path.join(root, set, "images")
            label_directory = os.path.join(root, set, "labels")

            output_image_directory = os.path.expanduser(f"{root}_augmented/{set}/images")
            output_label_directory = os.path.expanduser(f"{root}_augmented/{set}/labels")

            os.makedirs(output_label_directory, exist_ok=True)
            os.makedirs(output_image_directory, exist_ok=True)

            image_files = [f for f in os.listdir(image_directory) if f.endswith((".jpg", ".png", ".jpeg"))]
            label_files = [f for f in os.listdir(label_directory) if f.endswith(".txt")]

            if not image_files:
                print("Error: No image files found.")
                continue

            for image_file, label_file in tqdm(zip(image_files, label_files), total=len(image_files), desc="Processing"):
                img = cv2.imread(os.path.join(image_directory, image_file))
                augimg = img.copy()
                mode = random.choice([True, False])

                if mode:
                    bg_files = [f for f in os.listdir(bg_path) if f.endswith((".jpg", ".png", ".jpeg"))]
                    bg = cv2.imread(os.path.join(bg_path, bg_files[random.randrange(0, len(bg_files))]))
                    augimg, boxes = add_background_noise(augimg, background_img=bg, label_file=os.path.join(label_directory, label_file))
                    cv2.imwrite(os.path.join(output_image_directory, f"{image_file[:-4]}-augmented{image_file[-4:]}"), augimg)
                    for box in boxes:
                        with open(os.path.join(output_label_directory, f"{image_file[:-4]}-augmented.txt"), 'a') as f:
                            classes, x_center, y_center, width, height = box
                            f.write(f"{int(classes)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                else:
                    cv2.imwrite(os.path.join(output_image_directory, f"{image_file[:-4]}-augmented{image_file[-4:]}"), augimg)
                    shutil.copy(
                        src=os.path.join(label_directory, label_file),
                        dst=os.path.join(output_label_directory, f"{label_file[:-4]}-augmented{label_file[-4:]}"),
                    )
