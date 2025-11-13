# How to submit a new model to PrePATH

This document describes the recommended conventions and checklist for adding a new model implementation under `models/` so it can be used by the PrePATH toolkit.

## Overview

Contributors should add a model implementation file under `models/` (for example `models/my_model.py`) containing:

- a model builder function that returns a PyTorch `nn.Module` already moved to the requested `device` and set to `eval()`;
- a preprocessing transformer function that returns a `torchvision.transforms` pipeline for patch/image preprocessing.

Also register your model in `models/__init__.py` so the top-level `get_model` and `get_custom_transformer` functions can import and use it.

## Quick steps

1. Add `models/my_model.py` implementing `get_model` and `get_trans` (see the example below).
2. If your model needs a checkpoint, put it under `models/ckpts/` and add an entry in `__implemented_models` inside `models/__init__.py`.
3. Add import branches in `get_model` and `get_custom_transformer` in `models/__init__.py` so callers can use `get_model('my_model', ...)` and `get_custom_transformer('my_model')`.
4. Submit a PR with a short README or description, a small usage example, and local verification output.

## Recommended API (example)

Example file `models/my_model.py`:

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # define your model
        self.backbone = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


def get_model(device, gpu_num=1, jit=False):
    """Builds and returns a model moved to `device` and set to eval mode.

    Args:
        device (torch.device or str): target device, e.g. 'cuda' or 'cpu'
        gpu_num (int): number of GPUs (if you need to wrap with DataParallel)
        jit (bool): whether to return a JIT-traced model (optional)

    Returns:
        nn.Module: model on `device` in eval() mode
    """
    model = MyModel()
    model = model.to(device)
    model.eval()
    return model


def get_trans():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
```

Notes:
- `get_model` should return the model already moved to the given `device` and set to `eval()`.
- `get_trans` should return a torchvision-compatible transforms pipeline.

## Registering the model in `models/__init__.py`

1. Add your checkpoint mapping (if applicable) to the `__implemented_models` dict:

```python
__implemented_models['my_model'] = 'models/ckpts/my_model.pth'
```

2. In `get_model(...)` add a branch that imports and calls your `get_model`:

```python
elif model_name.lower() == 'my_model':
    from models.my_model import get_model as get_my_model
    model = get_my_model(device, gpu_num, jit=jit)
```

3. In `get_custom_transformer(...)` add a branch that imports and returns your transforms:

```python
elif model_name.lower() == 'my_model':
    from models.my_model import get_trans
    custom_trans = get_trans()
```

Follow the existing patterns in `models/__init__.py` for naming and lowercasing checks.

## PR checklist (recommended)

- [ ] Short README or docstring explaining model purpose and expected input size
- [ ] Example snippet showing how to call `get_model` and `get_custom_transformer` and run a single forward pass
- [ ] If a checkpoint is required, include its path and how it was produced (or share a private link to us)
- [ ] Notes about additional dependencies, environment or GPU requirements
- [ ] Local sanity check results: e.g., shape of output features for a sample patch

## Validation process (maintainers)

- When a PR is opened, maintainers will run a quick validation on a small internal leaderboard (a few datasets/tasks) to check that:
  - The model can be imported and built using `get_model`.
  - Preprocessing from `get_custom_transformer` runs without CPU bottlenecks.
  - Feature outputs have expected shapes and are reproducible.
  - Resource usage (memory / runtime) is reasonable for the reported configuration.

- If the model shows promise on the small leaderboard, maintainers will trigger a more comprehensive validation across all tasks/datasets and update the public/internal leaderboard accordingly.

## Notes & tips

- Keep the API surface stable: `get_model(device, gpu_num, jit=False)` and `get_trans()` help maintain consistent usage across the toolkit.
- Avoid heavy CPU-only preprocessing in transforms; use efficient torchvision operations or document if special handling is required.
- If your model needs a custom install step, document it in the PR and add a `requirements/...` entry if necessary.

Thank you for contributing — well-documented, small, reproducible changes get reviewed faster.