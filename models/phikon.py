from PIL import Image
import torch
from transformers import AutoImageProcessor, ViTModel


def get_phikon(device, gpu_num):
    processor = AutoImageProcessor.from_pretrained("owkin/phikon")
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False).to(device)
    model.eval()
    # if gpu_num > 1:
    #     model = torch.nn.parallel.DataParallel(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params/1000/1000)

    def func(image):
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        # get the features
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # (1, 768) shape
            return features
    return func