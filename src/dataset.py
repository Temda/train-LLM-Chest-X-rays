import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):

        self.data_frame = pd.read_csv(csv_file)
        
        self.img_dir = img_dir
        self.transform = transform
        
        self.disease_labels = self.data_frame.columns[1:]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.iloc[idx, 0]
        
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, FileNotFoundError):
            print(f"Error: Cannot open image {img_path}")
            raise

        labels = self.data_frame.iloc[idx, 1:].values.astype('float32')
        
        labels = torch.tensor(labels)

        if self.transform:
            image = self.transform(image)

        return image, labels

if __name__ == '__main__':
    from torchvision import transforms
    import matplotlib.pyplot as plt

    CSV_PATH = r"D:\train-LLM-Chest-X-rays\archive\new_labels.csv"
    IMG_DIR = r"D:\train-LLM-Chest-X-rays\archive\resized_images\resized_images"

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    try:
        dataset = ChestXrayDataset(csv_file=CSV_PATH, img_dir=IMG_DIR, transform=test_transform)
        print(f"✅ Dataset loaded successfully.")
        print(f"Total images: {len(dataset)}")
        print(f"Disease Classes ({len(dataset.disease_labels)}): {list(dataset.disease_labels)}")

        sample_idx = 0
        image, label = dataset[sample_idx]
        
        print("\n--- Sample Data Check ---")
        print(f"Image Shape: {image.shape}") # ต้องได้ (3, 224, 224)
        print(f"Label Tensor: {label}")
        
        print(f"Filename: {dataset.data_frame.iloc[sample_idx, 0]}")

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")