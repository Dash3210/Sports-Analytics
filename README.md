# Sports Analytics: A Hybrid Computer Vision Pipeline

This repository contains the source code for a comprehensive computer vision pipeline designed to perform multi-object detection, tracking, and semantic analysis on sports video footage. The project initially focused on football and was later adapted to demonstrate robust performance on out-of-domain field hockey data through a hybrid approach, combining deep learning with classical computer vision techniques.

## Key Features

  * **Multi-Object Detection**: Fine-tuned YOLOv8 model to detect and classify players, referees, goalkeepers, and the ball.
  * **Player Tracking & ID**: Integrated ByteTrack with a Kalman Filter to assign and maintain unique IDs for each player throughout the video.
  * **Jersey Number OCR**: An OCR module to read and identify player jersey numbers, with a temporal voting system for enhanced stability.
  * **Pitch Line Segmentation**: A U-Net model trained to segment and classify the geometric lines of a football pitch.
  * **Hybrid CV for Domain Adaptation**: A robust fallback system using motion detection (for the ball) and Hough Transforms (for lines) to handle new sports (e.g., field hockey) where the trained models are not applicable.
  * **Performance Optimization**: Implemented `supervision.InferenceSlicer` to improve detection accuracy on small or distant objects.

## Demo Showcase

Below is a demonstration of the full pipeline applied to a football match, showcasing object detection, tracking with unique IDs, jersey number recognition, and line segmentation overlays.

![Project Demo](./playback.gif)`)

## Tech Stack

The project leverages a modern stack of computer vision and deep learning libraries:

  * **Core Framework**: PyTorch
  * **Object Detection**: YOLOv11
  * **Computer Vision & Tracking**: OpenCV, `supervision`
  * **Optical Character Recognition**: EasyOCR
  * **Semantic Segmentation**: `segmentation-models-pytorch`
  * **Version Control**: Git, Git LFS

## Installation

To set up the environment and run this project locally, please follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Dash3210/Sports-Analytics.git
    cd Sports-Analytics
    ```

2.  **Install Git LFS:**
    This project uses Git LFS to manage large model files.

    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Install dependencies:**
    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Additional Assets:**
    The curated datasets and final HD output videos are available in the project's public Google Drive folder. Model weights are included in this repository via Git LFS.

## Usage

To run the full analysis pipeline on a video file, execute the main script from the terminal.

```bash
python run_pipeline.py --source_video "path/to/input.mp4" --output_video "path/to/output.mp4"
```

## Methodology

The pipeline processes video frames through a series of sequential modules to achieve a comprehensive analysis.

### 1\. Object Detection

A **YOLOv11** model was fine-tuned for the primary object detection task. The training utilized a curated version of the SoccerNet dataset, with initial contributions from [Mostafa-Nafie's repository](https://github.com/Mostafa-Nafie/Football-Object-Detection). The dataset underwent further alterations and cleaning to improve model performance on key classes like players and the ball.

### 2\. Line Segmentation

A **U-Net** architecture with a MobileNetV2 backbone was trained for semantic segmentation of the pitch lines. The training data was derived from the [SoccerNet Calibration dataset](https://github.com/SoccerNet/sn-calibration). The original 26 line classes were semantically grouped and curated down to 5 primary classes (e.g., penalty box, center circle, boundary lines) to create a more robust and generalized model.

### 3\. Player Tracking & OCR

Detected personnel are passed to a **ByteTrack** algorithm, which maintains a stable track ID for each individual. For tracked players, bounding box crops are extracted and passed to an **EasyOCR** module. A temporal voting system aggregates jersey number predictions across frames to produce a stable and accurate reading for each player ID.

### 4\. Domain Adaptation for Hockey

When applied to field hockey footage, the football-specific ball and line models failed due to **domain shift**. To solve this without retraining, a hybrid system was implemented:

  * **Ball Detection**: A classical motion detection algorithm based on background subtraction was used to identify the small, fast-moving hockey ball.
  * **Line Detection**: A color-based filter (HSV thresholding) combined with a **Hough Line Transform** was used to detect the white lines of the hockey pitch.

## Project Assets (Dataset & Outputs)

The complete set of project assets, including the curated datasets used for training, model weights, and final high-definition output videos for both football and hockey, are available for public access in the following Google Drive folder.

[**Link to Project Assets on Google Drive**](https://drive.google.com/drive/folders/1gNySNQhoUG_vc3BMze-HL-3S0LDm3XSV?usp=sharing)
