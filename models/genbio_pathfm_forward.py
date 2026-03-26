"""
https://huggingface.co/genbio-ai/genbio-pathfm
"""
from huggingface_hub import hf_hub_download
import torch
from torchvision import transforms
import os

def get_trans():
    """
    Returns preprocessing transforms for GenBio-PathFM.
    
    Uses custom normalization with 224x224 input size.
    """
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.697, 0.575, 0.728),
            std=(0.188, 0.240, 0.187)
        ),
    ])


def get_model(device):
    try:
        from genbio_pathfm.model import GenBio_PathFM_Inference
    except ImportError:
        raise ImportError(
            "In order to use GenBio-PathFM, please run the following: 'pip install git+https://github.com/genbio-ai/genbio-pathfm.git'"
        )

    ckpt_path = hf_hub_download(
        repo_id="genbio-ai/genbio-pathfm",
        filename="model.pth",
    )

    model = GenBio_PathFM_Inference(ckpt_path, device="cpu").to(device)

    model.eval()
    def func(img):
        # We recommend using mixed precision for faster inference.
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.inference_mode():
                features = model(img)
        return features
    return func

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    print(model)
    print(model(torch.rand((1, 3, 224, 224)).to(device
    )).shape)