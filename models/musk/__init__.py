from timm.models import create_model
from musk import utils, modeling
import torch
import torchvision
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


def get_model(device, jit=False):
    model = create_model("musk_large_patch16_384")
    utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')
    model.to(device=device, dtype=torch.float16)
    model.eval()
    def fn(img):
        with torch.inference_mode():
            image_embeddings = model(
                image=img.to(device, dtype=torch.float16),
                with_head=False,
                out_norm=True,
                ms_aug=True,
                return_global=True  
                )[0]  # return (vision_cls, text_cls)
        return image_embeddings
    return fn


def get_transform():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(384, interpolation=3, antialias=True),
        torchvision.transforms.CenterCrop((384, 384)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])
    return transform


if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    
    img = np.zeros((384, 384, 3), dtype=np.uint8)
    img = Image.fromarray(img)
    transform = get_transform()
    model = get_model("cuda")
    img = transform(img)
    img = img.unsqueeze(0)
    result = model(img)
    print(result.shape)
    print(result.dtype)