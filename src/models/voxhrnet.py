# from https://github.com/microsoft/VoxHRNet

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from yacs.config import CfgNode as CN


def convert_to_cfg_node(dictionary):
    return CN(dictionary)

def create_dropout_layer(p):
    return nn.Dropout(p=p)

def _replace_bn_relu(model):
    for name, module in model.named_children():
        # Replace BatchNorm3d with InstanceNorm3d
        if isinstance(module, nn.BatchNorm3d):
            setattr(model, name, nn.InstanceNorm3d(module.num_features, affine=True))
        # Replace ReLU with PReLU
        elif isinstance(module, nn.ReLU):
            setattr(model, name, nn.PReLU())
        else:
            # Recursively replace for submodules
            _replace_bn_relu(module)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, downsample=None, dropout=0.0):

        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.net = nn.Sequential(
            nn.Conv3d(in_planes, planes, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(planes),
            create_dropout_layer(dropout),
            nn.ReLU(),
            nn.Conv3d(planes, planes, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(planes),
            create_dropout_layer(dropout),
        )
        self.relu = nn.ReLU()

        return

    def forward(self, x):

        if self.downsample is None:
            residual = x
        else:
            residual = self.downsample(x)

        y = self.relu(self.net(x) + residual)

        return y


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_planes, planes, downsample=None, dropout=0.0):

        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.net = nn.Sequential(
            nn.Conv3d(in_planes, planes, 1, 1, 0, bias=False),
            nn.InstanceNorm3d(planes),
            create_dropout_layer(dropout),
            nn.ReLU(),
            nn.Conv3d(planes, planes, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(planes),
            create_dropout_layer(dropout),
            nn.ReLU(),
            nn.Conv3d(planes, planes * self.expansion, 1, 1, 0, bias=False),
            nn.InstanceNorm3d(planes * self.expansion),
            create_dropout_layer(dropout),
        )
        self.relu = nn.ReLU()

        return

    def forward(self, x):

        if self.downsample is None:
            residual = x
        else:
            residual = self.downsample(x)

        y = self.relu(self.net(x) + residual)

        return y


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class HighResolutionModule(nn.Module):

    def __init__(
        self,
        num_branches,
        block,
        num_blocks,
        num_inchannels,
        num_channels,
        multi_scale_output=True,
        dropout=0.0
    ):

        super(HighResolutionModule, self).__init__()
        self.num_branches = num_branches
        self.num_inchannels = num_inchannels
        self.dropout = dropout
        self.num_outchannels = [ele * block.expansion for ele in num_channels]
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(
            num_branches, block, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layer()
        self.relu = nn.ReLU()
        

    def _make_one_branch(self, block, num_block, num_inchannel, num_channel):

        if num_inchannel == num_channel * block.expansion:
            downsample = None
        else:
            downsample = nn.Sequential(
                nn.Conv3d(
                    num_inchannel, num_channel * block.expansion, 1, 1, 0, bias=False
                ),
                nn.InstanceNorm3d(num_channel * block.expansion),
                create_dropout_layer(self.dropout),
            )
        layers = [block(num_inchannel, num_channel, downsample, dropout=self.dropout)]
        layers.extend(
            [
                block(num_channel * block.expansion, num_channel, dropout=self.dropout)
                for i in range(num_block - 1)
            ]
        )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):

        branches = [
            self._make_one_branch(
                block, num_blocks[i], self.num_inchannels[i], num_channels[i]
            )
            for i in range(num_branches)
        ]

        return nn.ModuleList(branches)

    def _make_fuse_layer(self):

        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_outchannels = self.num_outchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    tem_net = nn.Sequential(
                        nn.Conv3d(
                            num_outchannels[j], num_outchannels[i], 1, 1, 0, bias=False
                        ),
                        nn.InstanceNorm3d(num_outchannels[i]),
                        create_dropout_layer(self.dropout),
                    )
                    fuse_layer.append(tem_net)
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        num_outchannel_3x3 = (
                            num_outchannels[i] if k == i - j - 1 else num_outchannels[j]
                        )
                        tem_net = [
                            nn.Conv3d(
                                num_outchannels[j],
                                num_outchannel_3x3,
                                3,
                                2,
                                1,
                                bias=False,
                            ),
                            nn.InstanceNorm3d(num_outchannel_3x3),
                            create_dropout_layer(self.dropout),
                        ]
                        if k != i - j - 1:
                            tem_net.append(nn.ReLU())
                        conv3x3s.append(nn.Sequential(*tem_net))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_outchannels(self):

        return self.num_outchannels

    def forward(self, x):

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        if self.num_branches == 1:
            return [x[0]]

        x_fuse = []
        for i, fuse_layer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_layer[0](x[0])
            for j in range(1, self.num_branches):
                if j > i:
                    y = y + F.interpolate(
                        fuse_layer[j](x[j]),
                        size=x[i].shape[-3:],
                        mode="trilinear",
                        align_corners=False,
                    )
                elif i == j:
                    y = y + x[j]
                else:
                    y = y + fuse_layer[j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


def create_dropout_layer(p):
    return nn.Dropout(p=p)


class HighResolutionNet(nn.Module):

    def __init__(self, config, in_channels=1, out_channels=7, dropout=0.0):

        super(HighResolutionNet, self).__init__()
        self.dropout = dropout
        self.num_classes = out_channels
        self.extra = config.MODEL.EXTRA
        self.stem_net = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, 2, 1, bias=False),
            nn.InstanceNorm3d(32),
            create_dropout_layer(self.dropout),
            nn.ReLU(),
            self._make_layer(Bottleneck, 32, 16, 2),
        )

        pre_stage_channels = [64]

        self.stage2_cfg = self.extra["STAGE2"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [
            num_channel * block.expansion
            for num_channel in self.stage2_cfg["NUM_CHANNELS"]
        ]
        self.transition1 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels
        )

        self.stage3_cfg = self.extra["STAGE3"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [
            num_channel * block.expansion
            for num_channel in self.stage3_cfg["NUM_CHANNELS"]
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels
        )

        if "STAGE4" in self.extra:
            self.stage4_cfg = self.extra["STAGE4"]
            block = blocks_dict[self.stage4_cfg["BLOCK"]]
            num_channels = [
                num_channel * block.expansion
                for num_channel in self.stage4_cfg["NUM_CHANNELS"]
            ]
            self.transition3 = self._make_transition_layer(
                pre_stage_channels, num_channels
            )
            self.stage4, pre_stage_channels = self._make_stage(
                self.stage4_cfg, num_channels
            )

        last_channels = int(np.sum(pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv3d(last_channels, last_channels, 1, 1, 0),
            nn.InstanceNorm3d(last_channels),
            create_dropout_layer(self.dropout),
            nn.ReLU(),
            nn.Conv3d(last_channels, self.num_classes, 1, 1, 0),
        )

    def _make_layer(self, block, in_planes, planes, num_block):

        if in_planes == planes * block.expansion:
            downsample = None
        else:
            downsample = nn.Sequential(
                nn.Conv3d(in_planes, planes * block.expansion, 1, 1, 0, bias=False),
                nn.InstanceNorm3d(planes * block.expansion),
            )
        layers = [block(in_planes, planes, downsample, dropout=self.dropout)]
        layers.extend(
            [block(planes * block.expansion, planes, dropout=self.dropout) for i in range(num_block - 1)]
        )

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre, num_channels_cur):

        num_branches_pre = len(num_channels_pre)
        num_branches_cur = len(num_channels_cur)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_pre[i] != num_channels_cur[i]:
                    tem_net = nn.Sequential(
                        nn.Conv3d(
                            num_channels_pre[i],
                            num_channels_cur[i],
                            3,
                            1,
                            1,
                            bias=False,
                        ),
                        nn.InstanceNorm3d(num_channels_cur[i]),
                        create_dropout_layer(self.dropout),
                        nn.ReLU(),
                    )
                    transition_layers.append(tem_net)
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                in_channels = num_channels_pre[-1]
                for j in range(i + 1 - num_branches_pre):
                    out_channels = (
                        num_channels_cur[i]
                        if j == i - num_branches_pre
                        else in_channels
                    )
                    tem_net = nn.Sequential(
                        nn.Conv3d(in_channels, out_channels, 3, 2, 1, bias=False),
                        nn.InstanceNorm3d(out_channels),
                        create_dropout_layer(self.dropout),
                        nn.ReLU(),
                    )
                    conv3x3s.append(tem_net)
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):

        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]

        modules = []
        for i in range(num_modules):
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    multi_scale_output or i != num_modules - 1,
                    self.dropout
                )
            )
            num_inchannels = modules[-1].get_num_outchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):

        original_shape = x.shape

        x = self.stem_net(x)

        x_list = [
            x if self.transition1[i] is None else self.transition1[i](x)
            for i in range(self.stage2_cfg["NUM_BRANCHES"])
        ]
        y_list = self.stage2(x_list)
        x_list = [
            (
                y_list[i]
                if self.transition2[i] is None
                else self.transition2[i](
                    y_list[i if i < self.stage2_cfg["NUM_BRANCHES"] else -1]
                )
            )
            for i in range(self.stage3_cfg["NUM_BRANCHES"])
        ]
        y_list = self.stage3(x_list)
        if "STAGE4" in self.extra:
            x_list = [
                (
                    y_list[i]
                    if self.transition3[i] is None
                    else self.transition3[i](
                        y_list[i if i < self.stage3_cfg["NUM_BRANCHES"] else -1]
                    )
                )
                for i in range(self.stage4_cfg["NUM_BRANCHES"])
            ]
            y_list = self.stage4(x_list)
        x = y_list

        for i in range(1, len(x)):
            x[i] = F.interpolate(
                x[i], size=x[0].shape[-3:], mode="trilinear", align_corners=False
            )

        x = torch.cat(x, 1)
        x = self.last_layer(x)
        x = F.interpolate(
            x, size=original_shape[-3:], mode="trilinear", align_corners=False
        )

        return x

    def set_weights(self, w):

        for idx, para in enumerate(self.parameters()):
            para.data = w[idx].cuda()

        return 0

    def get_parameters_copy(self):

        paras = [w.clone().detach() for w in self.parameters()]

        return paras
    
    
class VoxHRNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=7, dropout=0.0):
        super(VoxHRNet, self).__init__()
        
        model_config = {
            "MODEL": {
                "NAME": "voxhrnet",
                "EXTRA": {
                    "STAGE2": {
                        "NUM_MODULES": 1,
                        "NUM_BRANCHES": 2,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS": [3, 3],
                        "NUM_CHANNELS": [16, 32],
                    },
                    "STAGE3": {
                        "NUM_MODULES": 1,
                        "NUM_BRANCHES": 3,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS": [3, 3, 3],
                        "NUM_CHANNELS": [16, 32, 64],
                    },
                },
            }
        }


        model_config = convert_to_cfg_node(model_config)

        self.model = HighResolutionNet(
            model_config, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            dropout=dropout)
        
        _replace_bn_relu(self.model)

    def forward(self, image):
      
        logit = self.model(image)
        
        return logit
