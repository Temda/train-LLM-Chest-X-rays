import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm 
import os
import matplotlib.pyplot as plt

from dataset import ChestXrayDataset
from model import ChestXrayModel
from utils import calculate_metrics, save_checkpoint

BATCH_SIZE = 16        # ถ้าการ์ดจอแรมเยอะ ปรับเพิ่มเป็น 32 หรือ 64 ได้
LEARNING_RATE = 1e-4   # ค่ามาตรฐานสำหรับ Fine-tuning
NUM_EPOCHS = 10        # จำนวนรอบการเทรน
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_PATH = r"D:\train-LLM-Chest-X-rays\archive\new_labels.csv"
IMG_DIR = r"D:\train-LLM-Chest-X-rays\archive\resized_images\resized_images"
SAVE_DIR = "saved_models"

scaler = torch.amp.GradScaler('cuda') if DEVICE == "cuda" else None

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    
    loop = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item()) # อัปเดตค่า Loss ในหลอดโหลด
        
    return running_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval() 
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            all_preds.append(outputs)
            all_labels.append(labels)
            
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    metrics = calculate_metrics(all_labels, all_preds)
    metrics["loss"] = running_loss / len(loader)
    
    return metrics

def plot_metrics(metrics, save_dir, epoch):
    plt.figure(figsize=(10, 5))
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.title(f"Classification Metrics (Epoch {epoch})")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, f"classification_metrics_epoch_{epoch}.png"))
    plt.close()

def plot_f1_score(f1_scores, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(f1_scores) + 1), f1_scores, marker='o', color='green')
    plt.title("F1-Score Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, "f1_score_over_epochs.png"))
    plt.close()

def main():
    print(f"🚀 Starting Training on device: {DEVICE}")
    print(f"Using device: {DEVICE}")
    
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = ChestXrayDataset(CSV_PATH, IMG_DIR, transform=data_transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    
    print(f"Data Loaded: Train {len(train_dataset)} imgs, Val {len(val_dataset)} imgs")
    
    model = ChestXrayModel(num_classes=15, pretrained=True).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_f1 = 0.0
    f1_scores = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
        
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | F1-Macro: {val_metrics['f1_macro']:.4f} | AUC: {val_metrics['roc_auc']:.4f}")
        
        plot_metrics(val_metrics, SAVE_DIR, epoch + 1)
        
        f1_scores.append(val_metrics['f1_macro'])
        plot_f1_score(f1_scores, SAVE_DIR)
        
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            save_path = os.path.join(SAVE_DIR, "best_model_densenet.pth")
            save_checkpoint(model, optimizer, epoch, val_metrics, save_path)
            print(f"New Best F1! Model saved.")

if __name__ == "__main__":
    main()