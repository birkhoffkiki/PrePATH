import loki.utils
import torch
import torch.nn.functional as F


def get_model(device, ckpt_path):
    model, preprocess, _ = loki.utils.load_model(ckpt_path, device)
    model.eval()
    def fn(image_input):
        image_input = image_input.to(device)
        with torch.no_grad():
            image_embeddings = model.encode_image(image_input)
        # Normalize all embeddings across the feature dimension (L2 normalization)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        return image_embeddings
    
    return fn


def get_trans(ckpt_path):
    model, preprocess, _ = loki.utils.load_model(ckpt_path, 'cpu')
    del model
    return preprocess


if __name__ == '__main__':
    from PIL import Image
    p = '/jhcnas3/Pathology/code/PrePath/test_data/10082_30016_512_512.jpg'
    
    img = Image.open(p).convert('RGB')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # original implementation 
    model, preprocess, _ = loki.utils.load_model('/jhcnas3/Pathology/code/PrePath/models/ckpts/omiclip.pth', device)
    model.eval()
    image_embeddings = loki.utils.encode_images(model, preprocess, [p], device)
    
    # My implementation
    model = get_model(device, '/jhcnas3/Pathology/code/PrePath/models/ckpts/omiclip.pth')
    trans = get_trans('/jhcnas3/Pathology/code/PrePath/models/ckpts/omiclip.pth')
    my_image_embeddings = model(trans(img).unsqueeze(0))
    
    # error
    mae = torch.mean(torch.abs(image_embeddings - my_image_embeddings))
    print(f'MAE: {mae.item()}')
