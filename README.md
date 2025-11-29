# spoiled-pepperleaf-detector-classifier
YOLO-based Korean pepper leaf detection and ResNet18 disease classification for edge deployment

## Data Usage Notice

This project utilizes datasets provided by AI Hub (Korea Information Society Agency) for research and model development purposes.
According to AI Hub’s data usage policy, the original images, resized images, and any derived annotations cannot be redistributed or included in this repository.

The dataset used in this project was processed (resized to 640×640) and annotated using Grounding DINO and YOLO-based models, but these data and labels are not publicly released due to copyright and redistribution restrictions.

Only source code, configuration files, and model architectures are included here.
The trained model weights, original or processed images, and annotation files are excluded.

If you wish to reproduce the experiments, please:
	1.	Request and download the dataset directly from AI Hub after obtaining usage approval.
	2.	Place the downloaded data under the data/raw/ directory following the structure described in this repository.
	3.	Run the preprocessing and training scripts as provided.

### Labeling
Label from AI Hub was insufficient, so we used Grounding DINO for 1 stage labeling.
Then by roboflow, we corrected labels of 230 images by ourselves.
We trained yolov11s by 230 images and labeled leftover 160 images.
Then again by roboflow, we corrected labels of 160 images
Those labels for each stage are in directory. (Except raw label)

## Procedure

### Detection 
#### Download raw files
Download in AI hub

#### Select files for detection
run select_detection_subset.py

#### Resize images to 640 640


#### Test grounding DINO for labeling


#### Test yolov11s used for labeling

#### Test yolov8n and other models


### Classiciation
#### Download raw files
Download in AI hub





### Training

#### Train Detection (yolo)

#### Train classification (resnet)
run train_resent.py
python -m ai.classification.train_resnet --backbone resnet18/resent50


https://drive.google.com/drive/folders/13yicLG0txS0S_t1Iuv7l6Hv6CP9FztPP?usp=sharing, https://drive.google.com/drive/folders/1f9iVxkEr4StjCiHqpBWel3KNPRaKmPYh?usp=sharing, https://drive.google.com/drive/folders/1hWj7vHHJ99N5xNc6RJw6zf0Ag-FXeOsf?usp=sharing
