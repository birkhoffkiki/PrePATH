# https://huggingface.co/paige-ai/Virchow
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
    
    
def get_virchow_trans():
    model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    del model
    return transforms


def get_virchow_model(device):
    try:
        model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, 
                                mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU).to(device)
    except:
        print('Failed to load pretrained model from huggingface, loading from local')
        model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=False, 
                                mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU).to(device)
        msg = model.load_state_dict(torch.load('models/ckpts/virchow.pth'))
        print(msg)

    model.eval()
    def func(image):
        # get the features
        with torch.inference_mode():
            output = model(image)

        class_token = output[:, 0]    # size: 1 x 1280
        patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
        # concatenate class token and average pool of patch tokens
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
        return embedding # float32
    
    return func

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_virchow_model(device)
    img = torch.randn(1, 3, 224, 224).to(device)
    print(model(img).shape)
    print('Virchow model loaded successfully')
    