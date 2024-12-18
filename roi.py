import argparse
import time
from pathlib import Path

import cv2
import torch
from torch.backends import cudnn
from numpy import random
import os 

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging, increment_path
from utils.torch_utils import select_device


def detect_and_save_rois():
    source, weights, imgsz = opt.source, opt.weights, opt.img_size
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # print("파일저장 위치 : ",save_dir)

    save_dir = Path(os.path.join(os.getcwd(), 'ROIs',opt.name))
    save_dir.mkdir(parents=True, exist_ok=True)  # make directory for saving ROIs

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # warmup

    leads = []

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):
            p, im0 = Path(path), im0s.copy()
            save_path = str(save_dir / p.stem)  # Save path without extension

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                

                # Save ROIs and bounding box info
                txt_file_path = f"{save_path}.txt"
                with open(txt_file_path, 'w') as f:
                    for j, (*xyxy, conf, cls) in enumerate(det):

                        # Extract ROI
                        x1, y1, x2, y2 = map(int, xyxy)  # Convert to int
                        roi = im0[y1:y2, x1:x2]

                        # Write bounding box to TXT file (YOLO format)
                        x_center = (x1 + x2) / 2 / im0.shape[1]
                        y_center = (y1 + y2) / 2 / im0.shape[0]

                        width = (x2 - x1) / im0.shape[1]
                        height = (y2 - y1) / im0.shape[0]

                        # f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        # print(f"Saved bounding box info to {txt_file_path}")
                        roi_path = f"{save_path}_roi{j}.jpg"

                        leads.append({
                                    "classes":int(cls), 
                                    "x_center":round(x_center,6),
                                    "y_center":round(y_center,6), 
                                    "width":round(width,6), 
                                    "height":round(height,6), 
                                    "save_path":save_path,
                                    "roi":roi
                        })

                    
                        # Save ROI as an image file
                        # roi_path = f"{save_path}_roi{j}.jpg"
                        # cv2.imwrite(roi_path, roi)
                        # print(f"Saved ROI to {roi_path}")


    leads = sorted(
        leads, 
        key=lambda x: (x["classes"] != 1, x["x_center"] + (x["y_center"]*0.2) if x["classes"] == 1 else float('inf'))
    )

    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6','longII']
    for i, lead in enumerate(leads):

        for c, x, y, w, h, s, r in [list(lead.keys())]:
            
            classes = lead[c]
            x_center = lead[x]
            y_center = lead[y]
            width = lead[w]
            height = lead[h]
            save_path = lead[s]
            roi = lead[r]
            
            txt_file_path = f"{save_path}.txt"

            with open(txt_file_path, 'a') as f:
                f.write(f"{classes} {x_center} {y_center} {width} {height}\n")
                if len(leads) < 12 and i >= len(leads)-1:
                    roi_path = f"{save_path}_{lead_names[len(lead_names)-1]}_roi.jpg"
                else:
                    roi_path = f"{save_path}_{lead_names[i]}_roi.jpg"
                print(f"Saved bounding box info to {txt_file_path}")
                cv2.imwrite(roi_path, roi)
                print(f"Saved ROI to {roi_path}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    with torch.no_grad():
        detect_and_save_rois()
