import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from SoccerNet.utils import getListGames
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

# ==============================================================================
# 1. FINAL CONFIGURATION
# ==============================================================================
# Using raw strings for paths to avoid escape sequence errors
DATASET_PATH = r'D:\Image\SoccerNet'
MODEL_SAVE_PATH = r'D:\Image\best_semantic_line_model.pth'

TRAINING_RESOLUTION = (405, 720) # Using (height, width) consistently
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 6
EPOCHS = 50
LEARNING_RATE = 1e-4

# The complete class map - fixed non-printable character issue
LINE_CLASS_MAP = {
    "background": 0, "Big rect. left bottom": 1, "Big rect. left main": 1,
    "Big rect. left top": 1, "Big rect. right bottom": 1, "Big rect. right main": 1,
    "Big rect. right top": 1, "Small rect. left bottom": 1, "Small rect. left main": 1,
    "Small rect. left top": 1, "Small rect. right bottom": 1, "Small rect. right main": 1,
    "Small rect. right top": 1, "Circle central": 2, "Circle left": 2, "Circle right": 2,
    "Middle line": 2, "Goal left crossbar": 3, "Goal left post left": 3,
    "Goal left post right": 3, "Goal right crossbar": 3, "Goal right post left": 3,
    "Goal right post right": 3, "Side line bottom": 4, "Side line left": 4,
    "Side line right": 4, "Side line top": 4, "Goal unknown": 5, "Line unknown": 5
}
NUM_CLASSES = len(set(LINE_CLASS_MAP.values()))

# ==============================================================================
# 2. FINAL DATALOADER CLASS
# ==============================================================================
class SemanticLineDataset(Dataset):
    # CORRECTED: Use double underscores for special methods
    def __init__(self, path, split, resolution, class_map):
        self.path = path
        self.resolution = resolution
        self.class_map = class_map
        self.transform = T.Compose([T.ToTensor(), T.Resize(self.resolution, antialias=True)])

        full_list_games = getListGames(split, task="frames")
        self.list_games = [game for game in full_list_games if os.path.isdir(os.path.join(self.path, game))]
        
        self.samples = []
        for game in self.list_games:
            annotations_path = os.path.join(self.path, game, "Labels-v3.json")
            if not os.path.exists(annotations_path): continue
            with open(annotations_path) as f:
                annotations = json.load(f)
            all_entries = {**annotations.get("actions", {}), **annotations.get("replays", {})}

            for entry_name, entry_data in all_entries.items():
                image_path = os.path.join(self.path, game, "Frames-v3", entry_name)
                if os.path.exists(image_path) and entry_data.get("lines"):
                    self.samples.append({
                        "image_path": image_path, "lines": entry_data["lines"],
                        "original_resolution": (entry_data["imageMetadata"]["width"], entry_data["imageMetadata"]["height"])
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image_tensor = self.transform(image)
        
        mask = np.zeros(self.resolution, dtype=np.int32)
        orig_w, orig_h = sample["original_resolution"]

        # Define scaling factors
        scale_w = self.resolution[1] / orig_w
        scale_h = self.resolution[0] / orig_h

        for line_data in sample["lines"]:
            class_name = line_data.get("class", "").strip()
            class_id = self.class_map.get(class_name)
            if class_id and line_data.get("points"):
                points = np.array(line_data["points"]).reshape(-1, 2)
                # Scale points according to the resized image
                points[:, 0] = points[:, 0] * scale_w
                points[:, 1] = points[:, 1] * scale_h
                points = points.astype(int)
                cv2.polylines(mask, [points], isClosed=False, color=class_id, thickness=2)

        return image_tensor, torch.from_numpy(mask).long()

# ==============================================================================
# 3. MAIN EXECUTION BLOCK
# ==============================================================================
# CORRECTED: Main script wrapped to prevent multiprocessing errors
if __name__ == '__main__':
    # --- Setup Dataloaders ---
    train_dataset = SemanticLineDataset(path=DATASET_PATH, split='train', resolution=TRAINING_RESOLUTION, class_map=LINE_CLASS_MAP)
    valid_dataset = SemanticLineDataset(path=DATASET_PATH, split='valid', resolution=TRAINING_RESOLUTION, class_map=LINE_CLASS_MAP)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- Setup Model, Loss, and Optimizer ---
    model = smp.Unet("mobilenet_v2", encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES, activation=None)
    model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # CORRECTED: Removed deprecated 'verbose' parameter
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # --- Training & Validation Loop ---
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\n----- Epoch {epoch+1}/{EPOCHS} -----")
        
        # Training phase
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc="Training"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(valid_loader, desc="Validating"):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        
        print(f"Epoch {epoch+1} -> Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\n--- Training Complete ---")
    print(f"Best model saved to: {MODEL_SAVE_PATH} with validation loss: {best_val_loss:.4f}")