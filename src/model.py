import torch
import torch.nn as nn
from torchvision import models

class ChestXrayModel(nn.Module):
    def __init__(self, num_classes=15, pretrained=True):

        super(ChestXrayModel, self).__init__()
        
        if pretrained:
            weights = models.DenseNet121_Weights.DEFAULT
        else:
            weights = None
            
        self.backbone = models.densenet121(weights=weights)

        num_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x

if __name__ == '__main__':
    try:
        print("Init model...")
        model = ChestXrayModel(num_classes=15, pretrained=True)
        
        # ลองสร้างข้อมูลจำลอง (Dummy Input) ขนาด 1 รูป, 3 สี, 224x224
        dummy_input = torch.randn(1, 3, 224, 224)
        
        output = model(dummy_input)
        
        print("✅ Model created successfully!")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}") # ควรได้ [1, 15]
        
        if output.shape == (1, 15):
            print("Output dimension is correct (1 batch, 15 classes).")
        else:
            print("Output dimension is INCORRECT.")
            
    except Exception as e:
        print(f"Error: {e}")