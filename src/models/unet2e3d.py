from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.decoder import MyUnetDecoder3d  # noqa


def encode_for_resnet(e, x, B, depth_scaling=[2, 2, 2, 2, 1]):

    def pool_in_depth(x, depth_scaling):
        bd, c, h, w = x.shape
        x1 = x.reshape(B, -1, c, h, w).permute(0, 2, 1, 3, 4)
        x1 = F.avg_pool3d(
            x1,
            kernel_size=(depth_scaling, 1, 1),
            stride=(depth_scaling, 1, 1),
            padding=0,
        )
        x = x1.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        return x, x1

    encode = []
    x = e.conv1(x)
    x = e.bn1(x)
    x = e.act1(x)
    x, x1 = pool_in_depth(x, depth_scaling[0])
    encode.append(x1)
    # print(x.shape)
    # x = e.maxpool(x)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)

    x = e.layer1(x)
    x, x1 = pool_in_depth(x, depth_scaling[1])
    encode.append(x1)
    # print(x.shape)

    x = e.layer2(x)
    x, x1 = pool_in_depth(x, depth_scaling[2])
    encode.append(x1)
    # print(x.shape)

    x = e.layer3(x)
    x, x1 = pool_in_depth(x, depth_scaling[3])
    encode.append(x1)
    # print(x.shape)

    x = e.layer4(x)
    x, x1 = pool_in_depth(x, depth_scaling[4])
    encode.append(x1)
    # print(x.shape)

    return encode


class UNet2E3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 7,
        pretrained: bool = True
    ):
        super(UNet2E3D, self).__init__()
        self.output_type = [
            "infer",
            "loss",
        ]
        self.register_buffer("D", torch.tensor(0))

        arch = "resnet18d"
        encoder_dim = {
            "resnet18": [
                64,
                64,
                128,
                256,
                512,
            ],
            "resnet18d": [
                64,
                64,
                128,
                256,
                512,
            ],
        }.get(arch, [768])

        decoder_dim = [80, 80, 64, 32, 16]

        self.encoder = timm.create_model(
            model_name=arch,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            global_pool="",
            features_only=True,
        )

        self.decoder = MyUnetDecoder3d(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1] + [0],
            out_channel=decoder_dim,
        )
        self.mask = nn.Conv3d(decoder_dim[-1], out_channels, kernel_size=1)

    def forward(
        self,
        images: torch.Tensor,
    ) -> dict:
        images = images.squeeze(1)
        B, D, H, W = images.shape

        x = images.reshape(B * D, 1, H, W)
        x = x.expand(-1, 3, -1, -1)

        encode = encode_for_resnet(self.encoder, x, B, depth_scaling=[2, 2, 2, 2, 1])

        last, decode = self.decoder(
            feature=encode[-1],
            skip=encode[:-1][::-1] + [None],
            depth_scaling=[1, 2, 2, 2, 2],
        )

        logits = self.mask(last)

        return logits
