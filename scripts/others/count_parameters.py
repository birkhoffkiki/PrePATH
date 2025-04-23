import sys
sys.path.append('./')
from models import get_model


model = get_model('phikon', device='cpu', gpu_num=0)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params/1000/1000)


# import torch
# import torchvision.models as models

# # Load ResNet-50 (pretrained=False, num_classes=10 for example)
# model = models.resnet50(pretrained=False, num_classes=10)

# # Exclude the final fully connected layer (fc) from the parameter count
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad and not any(layer in str(p) for layer in ['fc']))

# print(f"Total parameters (excluding fc): {total_params}")