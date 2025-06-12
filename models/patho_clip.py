import torch
from PIL import Image
import open_clip


def get_model_ViT_L(device,ckpt_path):
    model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=ckpt_path)
    model.eval()
    model = model.to(device)
    
    def fn(image_tensor):
        with torch.inference_mode():
            image_embeddings = model.encode_image(image_tensor) 
        return image_embeddings
    return fn


def get_trans_ViT_L(ckpt_path):
    _, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=ckpt_path)
    return preprocess


if __name__ == '__main__':
    # test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = '/jhcnas3/Pathology/code/PrePath/models/ckpts/Patho-CLIP-L.pt'
    model = get_model_ViT_L(device, ckpt_path)
    preprocess = get_trans_ViT_L(ckpt_path)
    image = Image.open('/jhcnas3/Pathology/code/PrePath/test_data/10082_30016_512_512.jpg').convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    embeddings = model(image_tensor)
    
    print(embeddings.shape)  # Should print the shape of the embeddings
