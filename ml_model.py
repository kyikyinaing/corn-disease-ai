import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "corn_model.pth"

CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
NUM_CLASSES = 4

class CornDiseaseModel:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"{MODEL_PATH} not found in project folder.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build ResNet18 + replace FC
        base = models.resnet18(weights=None)
        base.fc = nn.Linear(base.fc.in_features, NUM_CLASSES)

        state = torch.load(MODEL_PATH, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        base.load_state_dict(state, strict=True)

        self.model = base.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict_from_image(self, image_path: str) -> dict:
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            conf = float(probs[idx].item())

        return {
            "label": CLASS_NAMES[idx],
            "confidence": conf,
            "all_probabilities": {CLASS_NAMES[i]: float(probs[i].item()) for i in range(NUM_CLASSES)}
        }