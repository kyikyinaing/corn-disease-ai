import torch
from torchvision import models, transforms
from PIL import Image

# 1) IMPORTANT: same order as your training labels
CLASS_NAMES = [
    "Healthy",
    "Gray Leaf Spot",
    "Common Rust",
    "Northern Leaf Blight",
]

MODEL_PATH = "corn_resnet18_best.pth"

# ImageNet normalization (typical for ResNet18)
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def load_model(device: str = "cpu"):
    model = models.resnet18(weights=None)   # your trained weights will be loaded
    model.fc = torch.nn.Linear(512, 4)      # 4 classes (confirmed from your .pth)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model

def predict_image(model, image_path: str, device: str = "cpu"):
    img = Image.open(image_path).convert("RGB")
    x = IMG_TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)

    return {
        "label_index": int(idx.item()),
        "label": CLASS_NAMES[int(idx.item())],
        "confidence": float(conf.item()),
        "all_probabilities": {CLASS_NAMES[i]: float(probs[i].item()) for i in range(4)}
    }