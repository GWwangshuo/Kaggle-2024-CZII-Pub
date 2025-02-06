import os
from glob import glob

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet
from onnxsim import simplify


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=7,
            channels=(48, 64, 80, 80),
            strides=(2, 2, 1),
            num_res_units=1,
        )

    def forward(self, x):
        logit = self.model(x)
        prob = F.softmax(logit, dim=1)
        return prob



ckpts= [
        'epoch155-step3744-valid_loss0.4729-val_metric0.7857.ckpt',
        'epoch178-step4296-valid_loss0.3901-val_metric0.8781.ckpt',
        'epoch130-step3144-valid_loss0.4365-val_metric0.7832.ckpt',
        'epoch198-step4776-valid_loss0.4278-val_metric0.7856.ckpt',
        'epoch80-step1944-valid_loss0.4681-val_metric0.8359.ckpt',
        'epoch184-step4440-valid_loss0.4091-val_metric0.8282.ckpt',
        'epoch180-step4344-valid_loss0.3825-val_metric0.8483.ckpt'
    ]
    
if __name__ == '__main__':
    
    model = Net()
    model.eval()
    
    root_dir = '/ssd/challenges/3D_UNET_MONAI/tb_logs.unet.256/TS_99_9/version_0/checkpoints'
    
    ckpt_paths = glob(f"{root_dir}/*.ckpt")
    
    for ckpt_path in ckpt_paths:
        vaild_id = ckpt_path.split('/')[5]
        ckpt_name = ckpt_path.split('/')[-1].replace('=', '')
        print(ckpt_name)
        
        if ckpt_name in ckpts:            
            state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
            print(model.load_state_dict(state_dict, strict=False))

            target_dir = f'./onnx_weights/{vaild_id}/'
            os.makedirs(target_dir, exist_ok=True)
            
            output_name = target_dir + ckpt_name.replace('ckpt', 'onnx')

            dummy_input = torch.rand((2, 1, 128, 384, 384))
            dynamic_axes = {"images": {0: "batch"}, "output": {0: "batch"}}

            torch.onnx.export(
                model,
                dummy_input,
                output_name,
                verbose=False,
                input_names=['images'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                opset_version=12
            )

            onnx_model = onnx.load(output_name)
            onnx.checker.check_model(onnx_model, full_check=True)
            model_simp, check = simplify(onnx_model)
            onnx.save(model_simp, output_name)