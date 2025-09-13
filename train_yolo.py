from torch import device
from ultralytics import YOLO

model = YOLO('yolo11s.pt')
model.train(
    data=f'D:/Image/SoccerNet/downloads/Dataset/data.yaml',
    batch=4,
    epochs=40,
    imgsz=1280,
    plots=True,
    hsv_v=0.8,
    lr0=0.03,
    scale=0.9,
    patience=8,
    device=device('cuda')
)
