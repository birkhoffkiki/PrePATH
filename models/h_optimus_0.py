"""
https://huggingface.co/bioptimus/H-optimus-0
"""
from huggingface_hub import login
import torch
import timm 
from torchvision import transforms


def get_trans():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.707223, 0.578729, 0.703617), 
            std=(0.211883, 0.230117, 0.177517)
        ),
    ])
    return transform


def get_model(device):
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False
    ).to(device)
    model.eval()
    def func(img):
        # We recommend using mixed precision for faster inference.
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.inference_mode():
                features = model(img)
        return features
    return func