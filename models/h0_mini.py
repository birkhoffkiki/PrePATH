"""
https://huggingface.co/bioptimus/H0-mini
"""
import timm
import torch
from huggingface_hub import login
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked


def get_trans():
    model = timm.create_model(
        "hf-hub:bioptimus/H0-mini", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
    )
    transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    del model
    return transforms

def get_model(device):
    model = timm.create_model(
        "hf-hub:bioptimus/H0-mini", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
    ).to(device)
    model.eval()

    def func(image):
        # get the features
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(image)
            class_token = output[:, 0]    # size: 1 x 768
            patch_tokens = output[:, 5:]  # size: 1 x 256 x 768, tokens 1-4 are register tokens so we ignore those
            # concatenate class token and average pool of patch tokens
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 1536
        return embedding # float32
    
    return func
