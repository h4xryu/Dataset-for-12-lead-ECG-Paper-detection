
import random
import shutil

import cv2
import os
import numpy as np

from scipy.stats import norm
import json

from tqdm import tqdm



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


def json2txt():
        for per, folder in enumerate(folders):
            print(f"processed : {per}/{len(folders)}")
            process_json_files(os.path.join(input_folder,folder))

def sample_datas():
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
                target_dir = os.path.join("/media/jwlee/9611c7a0-8b37-472c-8bbb-66cac63bc1c7/ECG_yolov7_datasets_v2")

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





if __name__ == "__main__":

    sample_datas()




