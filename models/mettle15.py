"""
METTLE 1.5 (Slideflow Labs): robustness fine-tune of H-optimus-0.
https://huggingface.co/slideflow-labs/mettle-1.5

Input : 224 x 224 RGB tile.
Output: 1536-dim refined CLS embedding (the model's default feature view).

Dependencies: torch>=2.2, timm>=1.0.15, transformers>=4.48, safetensors>=0.4.
The weights are public (not gated) and are downloaded from the Hugging Face Hub
on first use; no checkpoint is needed under models/ckpts/.
"""
import torch
from torchvision import transforms

MODEL_ID = "slideflow-labs/mettle-1.5"
# "cls": 1536-dim refined CLS embedding (default).
# "cls_mean": 3072-dim, refined CLS concatenated with the mean spatial patch token.
FEATURE_VIEW = "cls"


def get_trans():
    """Preprocessing from the model's preprocessor_config.json: direct bicubic
    resize to 224 x 224, scale to [0, 1], normalize."""
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517),
        ),
    ])


def get_model(device, gpu_num=1, jit=False):
    """Builds METTLE 1.5 on `device` in eval mode and returns a callable that
    maps a preprocessed batch (B, 3, 224, 224) to features (B, 1536)."""
    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError("METTLE 1.5 needs transformers: 'pip install \"transformers>=4.48\" \"timm>=1.0.15\" safetensors'")

    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    model.eval()
    use_amp = torch.device(device).type == "cuda"

    def func(img):
        # Mixed precision on GPU, as for the H-optimus-0 base model. The model
        # casts tokens to float32 before its CLS refinement head.
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                features = model(pixel_values=img, feature_view=FEATURE_VIEW).pooler_output
        return features.float()
    return func


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    print(model(torch.rand((1, 3, 224, 224)).to(device)).shape)
