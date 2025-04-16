import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms

class DummyManTraNet(torch.nn.Module):
    def __init__(self):
        super(DummyManTraNet, self).__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

def load_mantranet_model(weight_path):
    model = DummyManTraNet()
    model.load_state_dict(torch.load(weight_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def detect_forgery(model, image_np):
    image_resized = cv2.resize(image_np, (256, 256))
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(image_resized).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    heatmap = output.squeeze().numpy()
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    norm_heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    color_map = cv2.applyColorMap((norm_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return color_map
