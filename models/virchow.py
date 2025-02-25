# https://huggingface.co/paige-ai/Virchow
import timm
from torchvision import transforms
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from PIL import Image
    
    
def get_virchow_trans():
    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #     ]
    # )
    model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return transforms

def get_virchow_model(device):
    model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    # model = timm.create_model(
    #     "vit_huge_patch14_224_in21k", img_size=224, patch_size=14, init_values=1e-5, num_classes=0, dynamic_img_size=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
    # )
    # model.load_state_dict(torch.load('models/ckpts/.bin', map_location="cpu"), strict=True)
    # model.eval()
    model = model.eval()
    return model.to(device)