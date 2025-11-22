"""
OpenMidnight: A large vision transformer model for histopathology
https://huggingface.co/SophontAI/OpenMidnight

Based on Meta's DINOv2 ViT-G/14, fine-tuned for pathology images.
Outputs 1536-dimensional embeddings per patch.
"""

import torch
from huggingface_hub import hf_hub_download
from torchvision import transforms


def get_trans():
    """
    Returns preprocessing transforms for OpenMidnight.
    
    Uses standard ImageNet normalization with 224x224 input size.
    """
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])


def get_model(device):
    """
    Builds and returns the OpenMidnight model.
    
    Args:
        device (torch.device): Device to load model on (cuda or cpu)
    
    Returns:
        callable: Function that takes image tensor [B,3,H,W] and returns [B,1536] embeddings
    """
    # Download checkpoint from HuggingFace
    checkpoint_path = hf_hub_download(
        repo_id="SophontAI/OpenMidnight",
        filename="teacher_checkpoint_load.pt"
    )
    
    # Load base DINOv2 ViT-G/14 model
    model = torch.hub.load(
        'facebookresearch/dinov2',
        'dinov2_vitg14_reg',
        weights=None
    )

    # Load OpenMidnight weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Handle pos_embed parameter specially as per official usage
    pos_embed = checkpoint["pos_embed"]
    model.pos_embed = torch.nn.parameter.Parameter(pos_embed)
    model.load_state_dict(checkpoint)
    model.eval()
    model = model.to(device)
    
    def func(image):
        """
        Extract embeddings from image patches.
        
        Args:
            image: Tensor of shape [B, 3, 224, 224]
        
        Returns:
            Tensor of shape [B, 1536] - patch embeddings
        """
        with torch.inference_mode():
            # Use mixed precision for faster inference on CUDA
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    embeddings = model(image)
            else:
                embeddings = model(image)
        
        return embeddings  # [B, 1536]
    
    return func
