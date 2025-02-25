import torch
import timm
import numpy as np
from torchvision import transforms



__all__ = ['list_models', 'get_model', 'get_custom_transformer']


__implemented_models = {
    'ctranspath': 'models/ckpts/ctranspath.pth',
    'gpfm': 'models/ckpts/distill_87499.pth',
    'mstar': 'models/ckpts/mSTAR.pth'
}


def list_models():
    print('The following are implemented models:')
    for k, v in __implemented_models.items():
        print('{}: {}'.format(k, v))
    return __implemented_models


def get_model(model_name, device, gpu_num):
    """_summary_

    Args:
        model_name (str): the name of the requried model
        device (torch.device): device, e.g. 'cuda'
        gpu_num (int): the number of GPUs used in extracting features

    Raises:
        NotImplementedError: if the model name does not exist

    Returns:
        nn.Module: model
    """
    if model_name == 'resnet50':
        from models.resnet_custom import resnet50_baseline
        model = resnet50_baseline(pretrained=True).to(device)
        
    elif model_name.lower() == 'gpfm':
        from models.dinov2 import build_model
        model, _ = build_model(device, gpu_num, model_name, __implemented_models[model_name.lower()])

    elif model_name == 'ctranspath':
        from models.ctrans import ctranspath
        print('\n!!!! please note that ctranspath requires the modified timm 0.5.4, you can find package at here: models/ckpts/timm-0.5.4.tar , please install if needed ...\n')
        model = ctranspath(ckpt_path=__implemented_models['ctranspath']).to(device)
    
    elif model_name == 'plip':
        from models.plip import plip
        model = plip(device, gpu_num)
        
    elif model_name.lower() == 'uni':
        from models.uni import get_uni_model
        model = get_uni_model(device)

    elif model_name.lower() == 'conch':
        from models.conch import get_conch_model
        model = get_conch_model(device=device)
    
    elif model_name.lower() == 'mstar':
        from models.mSTAR import get_mSTAR_model
        model = get_mSTAR_model(device, __implemented_models[model_name.lower()])
        
    elif model_name == 'phikon':
        from models.phikon import get_phikon
        model = get_phikon(device, gpu_num)
        
    elif model_name == 'virchow':
        from models.virchow import get_virchow_model
        model = get_virchow_model(device)
        
    elif model_name == 'virchow2':
        from models.virchow2 import get_virchow_model
        model = get_virchow_model(device)
        
    elif model_name == 'gigapath':
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(device)
        model.eval()

    elif model_name == 'chief':
        from models.chief.ctran import get_model
        model = get_model(device=device)
        
    elif model_name.lower() == 'h-optimus-0':
        from models.h_optimus_0 import get_model
        model = get_model(device)

    else:
        raise NotImplementedError(f'{model_name} is not implemented')
    
    if model_name in ['resnet50', 'resnet101']:
        if gpu_num > 1:
            model = torch.nn.parallel.DataParallel(model)
        model = model.eval()
    
        
    return model


def get_custom_transformer(model_name):
    """_summary_

    Args:
        model_name (str): the name of model

    Raises:
        NotImplementedError: not implementated

    Returns:
        torchvision.transformers: the transformers used to preprocess the image
    """
    if model_name in ['resnet50', 'resnet101']:
        from models.resnet_custom import custom_transforms
        custom_trans = custom_transforms()
        
    elif model_name in ['phikon']:
        # Do nothing, let vit process do the image processing
        from torchvision import transforms as tt
        custom_trans = tt.Lambda(lambda x: torch.from_numpy(np.array(x)))
        
    elif model_name.lower() == 'uni':
        from models.uni import get_uni_trans
        custom_trans = get_uni_trans()
    
    elif model_name.lower() == 'conch':
        from models.conch import get_conch_trans
        custom_trans = get_conch_trans()
    
    elif model_name.lower() == 'mstar':
        from models.mSTAR import get_mSTAR_trans
        custom_trans = get_mSTAR_trans()
        
    elif model_name == 'virchow':
        from models.virchow import get_virchow_trans
        custom_trans = get_virchow_trans()
    
    elif model_name == 'virchow2':
        from models.virchow2 import get_virchow_trans
        custom_trans = get_virchow_trans()
    
          
    elif model_name == 'ctranspath':
        from models.ctrans import ctranspath_transformers
        custom_trans = ctranspath_transformers()
        
    elif model_name == 'plip':
        # Do nothing, let CLIP process do the image processing
        from torchvision import transforms as tt
        custom_trans = tt.Lambda(lambda x: torch.from_numpy(np.array(x)))
    
    elif model_name.lower() == 'gpfm':
        from models.dinov2 import build_transform
        custom_trans = build_transform()
        
    elif model_name == 'gigapath':
        custom_trans = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        
    elif model_name == 'chief':
        from models.chief.ctran import get_trans
        custom_trans = get_trans()
        
    elif model_name.lower() == 'h-optimus-0':
        from models.h_optimus_0 import get_trans
        custom_trans = get_trans()
        
    else:
        raise NotImplementedError('Transformers for {} is not implemented ...'.format(model_name))

    return custom_trans
