import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import os

def calculate_metrics(y_true, y_pred_logits, threshold=0.5):
    y_pred_probs = torch.sigmoid(y_pred_logits).detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    
    y_pred_binary = (y_pred_probs > threshold).astype(int)
    
    f1_macro = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    
    f1_micro = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    
    acc = accuracy_score(y_true, y_pred_binary)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred_probs, average='macro')
    except:
        roc_auc = 0.0
        
    return {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "accuracy": acc,
        "roc_auc": roc_auc
    }

def save_checkpoint(model, optimizer, epoch, metrics, filename="checkpoint.pth"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filename)
    print(f"💾 Saved checkpoint: {filename}")