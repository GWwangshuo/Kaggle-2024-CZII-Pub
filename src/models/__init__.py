from monai.networks.nets import DynUNet, SegResNet, UNet

from .densevnet import DenseVNet
from .unet2e3d import UNet2E3D
from .voxhrnet import VoxHRNet
from .voxresnet import VoxResNet

class_mapping = {
    'monai_unet': UNet,
    'segresnet': SegResNet,
    'dynunet': DynUNet,
    'densevnet': DenseVNet,
    'unet2e3d': UNet2E3D,
    'voxhrnet': VoxHRNet,
    'voxresnet': VoxResNet
}

def create_model(model_name, **parameters):

    if model_name in class_mapping:
        model = class_mapping[model_name](**parameters)
        model.out_channels = parameters["out_channels"]
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model