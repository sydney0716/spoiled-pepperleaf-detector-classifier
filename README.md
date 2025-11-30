# spoiled-pepperleaf-detector-classifier
YOLO-based Korean pepper leaf detection and ResNet18 disease classification for edge deployment.

## Data Usage Notice

This project utilizes datasets provided by AI Hub (Korea Information Society Agency) for research and model development purposes.
According to AI Hub’s data usage policy, the original images, resized images, and any derived annotations cannot be redistributed or included in this repository.

The dataset used in this project was processed (resized to 640×640) and annotated using Grounding DINO and YOLO-based models, but these data and labels are not publicly released due to copyright and redistribution restrictions.

Only source code, configuration files, and model architectures are included here.
The trained model weights, original or processed images, and annotation files are excluded.

If you wish to reproduce the experiments, please:
1. Request and download the dataset directly from AI Hub after obtaining usage approval.
2. Place the downloaded data under the `data/` directory following the structure described in this repository.
3. Run the preprocessing and training scripts as provided.

## Setup

### 1. Environment
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Download Pretrained Models
Download the base weights required for training and auto-labeling into `models/pretrained/`.

**GroundingDINO (Swin-T) & Config** (Used for Auto-Labeling):
```bash
mkdir -p models/pretrained
wget -P models/pretrained https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -P models/pretrained https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
```

**YOLO Base Models** (v8n, v8s, v11n, v11s):
```bash
# YOLOv8 Nano
wget -P models/pretrained https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
# YOLOv8 Small
wget -P models/pretrained https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt
# YOLO11 Nano
wget -P models/pretrained https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
# YOLO11 Small
wget -P models/pretrained https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
```

## Labeling Workflow
The labeling from AI Hub was insufficient, so we adopted a semi-automated pipeline:
1. **Grounding DINO**: Used for initial 1st-stage labeling.
2. **Manual Correction**: Corrected ~230 images using Roboflow.
3. **YOLOv11s**: Trained on the corrected 230 images to label the remaining ~160 images.
4. **Final Correction**: Manually verified the remaining labels.

### Labeling Examples
![Labeling with Grounding DINO](assets/labeling_groundingDINO.png)
*Grounding DINO Labeling*

![Labeling with YOLOv11s](assets/labeling_yolov11s.png)
*YOLOv11s Labeling*

## Procedure

### Detection Pipeline

1. **Download Data**
   - Download the dataset archive (zip) which contains the pre-processed data.
   - *Link:* [Processed Dataset](https://drive.google.com/file/d/1rpqLOnmaRJqQnmmzvI4QcXY52M85mHnL/view?usp=share_link)
   - **Structure:** The zip file contains three directories:
     - `classification_processed/`: Ready for ResNet training.
     - `detection_processed/`: Ready for YOLO training.
     - `labeling_data_for_labeling_yolo11s/`: Intermediate data for the labeling pipeline.
   - **Action:** Unzip the contents into the `data/` directory.
     ```bash
     unzip <downloaded_file>.zip -d data/
     ```

2. **Select Files**
   - Select files for detection (`select_detection_subset.py`).

3. **Preprocessing**
   - Resize images to 640x640.

4. **Run Labeling**
   - Test Grounding DINO for labeling.
   - Run `data_prep/run_labeling_grounding_dino.py`.

5. **Train/Test YOLO**
   - Train YOLOv11s on the initial set.
   - Run `training/train_yolo.py`.

### Detection Results
![YOLO Comparison](assets/yolo_comparison_plot.png)

### Classification Pipeline

1. **Download Raw Data**
   - Download classification dataset from AI Hub.

2. **Train ResNet**
   ```bash
   # Train ResNet18 (default)
   python training/train_resnet.py --backbone resnet18
   
   # Train ResNet50
   python training/train_resnet.py --backbone resnet50
   ```

### Classification Results
![ResNet Comparison](assets/resnet_comparison.png)