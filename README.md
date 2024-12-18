# 12-Lead ECG Dataset for YOLOv7

This repository provides resources and scripts to prepare and augment 12-lead ECG datasets for use with YOLOv7 models. The scripts include utilities for crawling background images, augmenting ECG data with various effects, extracting Regions of Interest (ROIs), and managing dataset organization.

***

### Examples

![nn](https://ifh.cc/g/lQGQhP.png)
![nn](https://ifh.cc/g/tQaFhh.png)

*** 
## Requirements
To run the scripts in this repository, install the following dependencies:


### Install Requirements
To install the dependencies on Ubuntu, use the following command:
```bash
pip install -r requirements.txt
```

Ensure you have Python and `pip` installed on your system. You can verify this with:
```bash
python --version
pip --version
```
***

## Contents

### `crawling_bg.py`
This script is used to download background images for data augmentation. It utilizes the `icrawler` library to fetch images from Google.

#### Function: `crawl_image_data`
```python
def crawl_image_data(key_word: str, max_num: int, des_dir: str):
    from icrawler.builtin import GoogleImageCrawler
    google_crawler = GoogleImageCrawler(storage={'root_dir': des_dir})
    google_crawler.crawl(keyword=key_word, max_num=max_num)
```

- **Parameters:**
  - `key_word`: The keyword to search for background images.
  - `max_num`: Maximum number of images to download.
  - `des_dir`: Directory to save the downloaded images.

#### Usage Example:
```python
if __name__ == '__main__':
    key_word = 'top view table background'
    max_num = 10000
    des_dir = './bg_noises'
    crawl_image_data(key_word, max_num, des_dir)
```

### `augmentation.py`
This script provides functions for augmenting ECG images and labels by adding background noise, applying shadows, and introducing geometric distortions.

#### Global Variables:
- `rectangles`: List to store rectangle coordinates.
- `start_point`, `end_point`: Points for drawing rectangles.
- `drawing`, `delete_mode`: Flags for drawing and deletion modes.
- `image`, `img_copy`, `img_with_text`: Variables for the working image.

#### Key Functions:

##### 1. `add_background_noise`
Adds background noise to an image and adjusts bounding box labels accordingly.

- **Highlights:**
  - Pads the image with random borders.
  - Adds background noise using a provided background image.
  - Adjusts bounding box coordinates in YOLO format.

##### 2. `generate_spot_light_mask`
Generates a spotlight effect mask with Gaussian decay.

- **Parameters:**
  - `mask_size`: Size of the mask.
  - `position`: Center of the spotlight.
  - `max_brightness`, `min_brightness`: Brightness range.
  - `mode`: Decay mode (Gaussian or linear).
  - `speedup`: Optimization flag.

##### 3. `add_spot_light`
Applies the spotlight effect to an image.

- **Parameters:**
  - `image`: Input image.
  - `light_position`: Center of the spotlight.
  - `transparency`: Blend ratio for the effect.

##### 4. `add_bend_distortion`
Applies bend distortion to an image using sinusoidal transformations.

#### Dataset Augmentation Workflow
The script processes images and labels, applying augmentations with background noise and other effects. The augmented data is saved in designated directories.

#### Example Workflow:
```python
if __name__ == "__main__":
    for set in set_list:
        print(f"Current set: {set}")
        image_directory = os.path.join(root, set, "images")
        label_directory = os.path.join(root, set, "labels")
        output_image_directory = os.path.join(root + "_augmented", set, "images")
        output_label_directory = os.path.join(root + "_augmented", set, "labels")

        os.makedirs(output_image_directory, exist_ok=True)
        os.makedirs(output_label_directory, exist_ok=True)

        for image_file, label_file in tqdm(zip(image_files, label_files), total=len(image_files), desc="Processing"):
            img = cv2.imread(os.path.join(image_directory, image_file))
            aug_img, boxes = add_background_noise(img, background_img, label_file)
            cv2.imwrite(os.path.join(output_image_directory, f"{image_file[:-4]}-augmented{image_file[-4:]}"), aug_img)
            # Save adjusted labels
            with open(os.path.join(output_label_directory, f"{label_file[:-4]}-augmented.txt"), 'w') as f:
                for box in boxes:
                    f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
```

### `roi.py`
This script extracts Regions of Interest (ROIs) from images using a YOLOv7 model and saves them as individual image files along with their bounding box information.

#### Key Function: `detect_and_save_rois`
- Extracts ROIs from input images based on YOLOv7 detections.
- Saves ROIs as separate image files and bounding box information in YOLO format.
- Automatically organizes and sorts the leads based on classes and positions.

#### Usage:
Run the script with the following command:
```bash
python roi.py --weights [pt_file] --source [image_file]
```

#### Arguments:
- `--weights`: Path to the YOLOv7 model file (e.g., `yolov7.pt`).
- `--source`: Path to the image or folder of images for inference.

***

## Notes
- Ensure the `icrawler`, `cv2`, and `torch` libraries are installed before running the scripts.
- The `augmentation.py` script assumes images and labels follow the YOLO format.
- Background images should be downloaded using `crawling_bg.py` and stored in the `bg_noises` directory.

