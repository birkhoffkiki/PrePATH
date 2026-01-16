"""
Replace otsu thresholding with deep learning based segmentation model.
Modified from https://github.com/mahmoodlab/TRIDENT/blob/main/trident/segmentation_models/load.py
"""
import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from huggingface_hub import hf_hub_download
import numpy as np


def get_weights_path():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ckpt_dir = os.path.join(dir_path, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    segmentation_model_name = 'deeplabv3_seg_v4.ckpt'
    weights_path = os.path.join(ckpt_dir, segmentation_model_name)
    if not os.path.exists(weights_path):
        # Download `deeplabv3_seg_v4.ckpt` from https://huggingface.co/MahmoodLab/hest-tissue-seg/tree/main
        repo_id = 'MahmoodLab/hest-tissue-seg'
        weights_path = hf_hub_download(repo_id=repo_id, filename=segmentation_model_name, cache_dir=ckpt_dir)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Segmentation model weights not found at {weights_path}")
    return weights_path


class SegmentationModel():
    def __init__(self, confidence_thresh=0.5, batch_size=24, overlap=32):
        self.confidence_thresh = confidence_thresh
        self.batch_size = batch_size
        self.overlap = overlap
        weights_path = get_weights_path()

        # Initialize base model
        model = deeplabv3_resnet50(weights=None)
        model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1, stride=1)
        # Load and clean checkpoint
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = {
            k.replace('model.', ''): v
            for k, v in checkpoint.get('state_dict', {}).items()
            if 'aux' not in k
        }

        msg = model.load_state_dict(state_dict)
        print(f"Segmentation model loaded with message: {msg}")
        # Store configuration
        self.input_size = 512
        self.precision = torch.float16
        self.target_mag = 10

        self.eval_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        model.eval()
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    @torch.no_grad()
    def predict(self, image):
        # input should be of shape (batch_size, C, H, W)
        logits = self.model(image)['out']
        softmax_output = F.softmax(logits, dim=1)
        predictions = (softmax_output[:, 1, :, :] > self.confidence_thresh).to(torch.uint8)  # Shape: [bs, 512, 512]
        return predictions
    
    def segment(self, large_image: np.array):
        """
        Segment a large image using the deep learning model with overlapping patches.
        Args:
            large_image (np.array): Large input image of shape (H, W, C).
        Returns:
            np.array: Segmentation mask of shape (H, W) with binary values.
        """
        H, W, C = large_image.shape
        stride = self.input_size - self.overlap
        # crop large image into overlapping patches
        patches = []
        positions = []
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y1 = y
                y2 = min(y + self.input_size, H)
                x1 = x
                x2 = min(x + self.input_size, W)
                patch = large_image[y1:y2, x1:x2, :]
                # Pad patch if it's smaller than input_size
                pad_bottom = self.input_size - (y2 - y1)
                pad_right = self.input_size - (x2 - x1)
                # pad
                patch = np.pad(patch, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='constant', constant_values=0)
                
                # Apply eval transforms
                patch = self.eval_transforms(patch)
                patches.append(patch)
                positions.append((y1, y2, x1, x2))
        
        patches = torch.stack(patches)  # Shape: [num_patches, C, input_size, input_size]
        
        # predict in batches
        all_predictions = []
        for i in range(0, len(patches), self.batch_size):
            with torch.autocast(device_type=self.device.type, dtype=self.precision):
                batch_patches = patches[i:i+self.batch_size].to(self.device, dtype=self.precision)
                batch_predictions = self.predict(batch_patches)
                all_predictions.append(batch_predictions.cpu().numpy())
        
        # Combine all predictions
        segmentation_mask = np.zeros((H, W), dtype=np.uint8)
        count_mask = np.zeros((H, W), dtype=np.uint8)
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        for idx, (y1, y2, x1, x2) in enumerate(positions):
            pred_patch = all_predictions[idx]
            h_patch = min(y2 - y1, self.input_size)
            w_patch = min(x2 - x1, self.input_size)
            segmentation_mask[y1:y1+h_patch, x1:x1+w_patch] += pred_patch[:h_patch, :w_patch]
            count_mask[y1:y1+h_patch, x1:x1+w_patch] += 1
        
        # Avoid division by zero
        count_mask[count_mask == 0] = 1
        segmentation_mask = segmentation_mask / count_mask
        segmentation_mask = (segmentation_mask > self.confidence_thresh).astype(np.uint8)*255
        return segmentation_mask


if __name__ == "__main__":
    # Test the segmentation model
    from openslide import OpenSlide
    import matplotlib.pyplot as plt
    

    # Load a sample image
    handle = OpenSlide('/jhcnas5/Pathology/original_data/Gastric/PWH/Stomach_Biopsy/Stomach_Biopsy/Stomach Biopsy(mrxs)/23S005992-I.mrxs')
    
    level = handle.get_best_level_for_downsample(64)
    image = handle.read_region((0, 0), level, handle.level_dimensions[level]).convert("RGB")
    image_np = np.array(image)

    # Initialize segmentation model
    seg_model = SegmentationModel(confidence_thresh=0.5, batch_size=4, overlap=32)

    # Perform segmentation
    seg_mask = seg_model.segment(image_np)

    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image_np)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Segmentation Mask')
    plt.imshow(seg_mask, cmap='gray')
    plt.axis('off')

    # save the figure
    plt.savefig('segmentation_result.png')