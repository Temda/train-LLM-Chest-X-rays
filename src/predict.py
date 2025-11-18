import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from model import ChestXrayModel

MODEL_PATH = r"saved_models/best_model_densenet.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

def disable_inplace_relu(model):

    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False

def load_trained_model():
    print(f"🔄 Loading model from: {MODEL_PATH}")
    model = ChestXrayModel(num_classes=len(CLASS_NAMES), pretrained=False)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()
    
    disable_inplace_relu(model)
    
    return model

def generate_grad_cam(model, input_tensor, original_image, target_class_index):
    feature_maps = []
    gradients = []

    def forward_hook(module, input, output):
        feature_maps.append(output.clone().detach())

    def backward_hook(module, grad_in, grad_out):
        if grad_out[0] is not None:
            gradients.append(grad_out[0].detach())

    target_layer = model.backbone.features
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad()
        output_logits = model(input_tensor)
        
        target_logit = output_logits[0][target_class_index]
        target_logit.backward()

        if not gradients or not feature_maps:
            print("Warning: No gradients or feature maps captured")
            return np.zeros((original_image.height, original_image.width))

        grads = gradients[0].cpu().numpy()[0]
        fmaps = feature_maps[0].cpu().numpy()[0]
        
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(fmaps.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * fmaps[i, :, :]

        cam = np.maximum(cam, 0)
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        cam = cv2.resize(cam, (original_image.width, original_image.height))
        
    except Exception as e:
        print(f"Error in Grad-CAM generation: {e}")
        cam = np.zeros((original_image.height, original_image.width))
    finally:
        forward_handle.remove()
        backward_handle.remove()
    
    return cam

def predict_image(model, image_path, target_class_name=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        input_tensor = input_tensor.to(DEVICE)
    except Exception as e:
        print(f"❌ Error opening image: {e}")
        return None, None, None, None

    with torch.no_grad():
        output_logits = model(input_tensor)
        output_probs = torch.sigmoid(output_logits)
        probs = output_probs.cpu().detach().numpy()[0]

    if target_class_name and target_class_name in CLASS_NAMES:
        target_class_index = CLASS_NAMES.index(target_class_name)
    else:
        target_class_index = np.argmax(probs)
        target_class_name = CLASS_NAMES[target_class_index]
        print(f"🔬 Generating Grad-CAM for the highest probability class: '{target_class_name}'")

    grad_input = input_tensor.clone().detach().requires_grad_(True)
    heatmap = generate_grad_cam(model, grad_input, image, target_class_index)

    return image, probs, heatmap, target_class_name

def apply_heatmap(image, heatmap):
    """ สร้างภาพ overlay heatmap บนภาพ gốc """
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed_img = heatmap_colored * 0.4 + np.array(image)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

def visualize_result(image, probs, heatmap, target_class_name):
    superimposed_img = apply_heatmap(image, heatmap)

    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Input X-ray")
    
    plt.subplot(1, 3, 2)
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title(f"Grad-CAM for '{target_class_name}'")

    plt.subplot(1, 3, 3)
    y_pos = np.arange(len(CLASS_NAMES))
    
    colors = ['red' if p > 0.5 else 'skyblue' for p in probs]
    
    plt.barh(y_pos, probs, align='center', color=colors)
    plt.yticks(y_pos, CLASS_NAMES)
    plt.xlabel('Probability (0-1)')
    plt.title('Disease Prediction')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(probs):
        plt.text(v + 0.01, i, f"{v:.2f}", va='center', fontweight='bold' if v > 0.5 else 'normal')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    TEST_IMAGE_PATH = r"D:\train-LLM-Chest-X-rays\predict_images\test.png"
    TARGET_DISEASE_FOR_CAM = None
    
    if os.path.exists(TEST_IMAGE_PATH):
        model = load_trained_model()
        print(f"🔍 Analyzing image: {TEST_IMAGE_PATH} ...")
        
        img, probabilities, heatmap, target_class = predict_image(
            model, 
            TEST_IMAGE_PATH, 
            target_class_name=TARGET_DISEASE_FOR_CAM
        )
        
        if img is not None:
            print("\n--- Prediction Results ---")
            for name, prob in zip(CLASS_NAMES, probabilities):
                status = "[/] POSITIVE" if prob > 0.5 else "   Negative"
                print(f"{name:20s}: {prob:.4f}  {status}")
                
            visualize_result(img, probabilities, heatmap, target_class)
    else:
        print(f"Image not found: {TEST_IMAGE_PATH}")
