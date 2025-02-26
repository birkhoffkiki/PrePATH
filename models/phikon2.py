from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel


# Load phikon-v2

def get_model(device):
    processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    model = AutoModel.from_pretrained("owkin/phikon-v2").to(device)
    model.eval()
    
    def func(image):
        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        # Get the features
        with torch.inference_mode():
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # (1, 1024) shape
        return features
    return func

