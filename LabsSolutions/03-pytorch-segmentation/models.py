#!/usr/bin/env python3
"""
    This script belongs to the lab work on semantic segmenation of the
    deep learning lectures https://github.com/jeremyfix/deeplearning-lectures
    Copyright (C) 2022 Jeremy Fix

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# Standard imports
import sys
import logging

# External imports
import torch
import torch.nn as nn
import torchvision.models

available_models = ["fcn_resnet50", "UNet"]


class TorchvisionModel(nn.Module):
    def __init__(self, modelname, input_size, num_classes, pretrained, in_channels):
        super().__init__()
        if pretrained:
            logging.info("Loading a pretrained model")
        else:
            logging.info("Loading a model with random init")
        exec(
            f"self.model = torchvision.models.segmentation.{modelname}(pretrained={pretrained}, pretrained_backbone={pretrained})"
        )
        old_conv1 = self.model.backbone.conv1
        self.model.backbone.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.stride,
        )
        old_head = self.model.classifier
        last_conv = list(old_head.children())[-1]
        self.model.classifier = nn.Sequential(
            *list(old_head.children())[:-1],
            nn.Conv2d(
                in_channels=last_conv.in_channels,
                out_channels=num_classes,
                kernel_size=last_conv.kernel_size,
                stride=last_conv.stride,
                padding=last_conv.stride,
            ),
        )

    def forward(self, x):
        return self.model(x)["out"]


def fcn_resnet50(input_size, num_classes):
    return TorchvisionModel("fcn_resnet50", input_size, num_classes, True, 3)


class UNetConvBlock(nn.Module):
    def __init__(self, num_inputs, num_channels):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_inputs,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs):
        features = self.block2(self.block1(inputs))
        outputs = self.block3(features)
        return outputs, features


class UNetEncoder(nn.Module):
    def __init__(self, num_inputs, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList()
        num_channels = 32
        for i in range(num_blocks):
            self.blocks.append(UNetConvBlock(num_inputs, num_channels))
            # Prepare the parameters for the next layer
            num_inputs = num_channels
            num_channels = 2 * num_channels

        # Add the last encoding layer
        self.last_block = nn.Sequential(
            nn.Conv2d(num_inputs, num_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        prev_outputs, lfeatures = inputs, []
        for b in self.blocks:
            outb, featb = b(prev_outputs)
            # Keep track of the encoder features
            # to be given to the decoder pathway
            lfeatures.append(featb)
            # Prepare the input for the next block
            prev_outputs = outb
        outputs = self.last_block(prev_outputs)
        # Here :
        # outputs is the output tensor of the last encoding layer
        # lfeatures is the output features of the num_blocks blocks
        return outputs, lfeatures


class UNetUpConvBlock(nn.Module):
    def __init__(self, num_inputs, num_channels):
        super().__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # TODO: the following is said, in the paper
            # to be a Conv (2x2) but the arithmetics lead to incorrect
            # shapes e.g. with encoder features 32x32, we get a map 33x33
            # with kernel_size=2, stride=1, padding=0
            nn.Conv2d(
                in_channels=num_inputs,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.convblock = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs, encoder_features):
        inter_features = self.upconv(inputs)  # B, C, H, W
        concat_features = torch.cat((encoder_features, inter_features), dim=1)
        # concatenate inter_features and encoder_features
        outputs = self.convblock(concat_features)
        return outputs


class UNetDecoder(nn.Module):
    def __init__(self, num_inputs, num_blocks, num_classes):
        super().__init__()
        num_inputs = 32 * 2**num_blocks
        num_channels = num_inputs
        self.first_block = nn.Sequential(
            nn.Conv2d(num_inputs, num_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        num_inputs = num_channels
        num_channels = num_channels // 2
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(UNetUpConvBlock(num_inputs, num_channels))
            # Prepare the parameters for the next layer
            num_inputs = num_channels
            num_channels = num_channels // 2

        # Add the last encoding layer
        self.last_conv = nn.Conv2d(
            num_inputs, num_classes, kernel_size=1, stride=1, padding=0
        )

    def forward(self, encoder_outputs, encoder_features):
        outputs = self.first_block(encoder_outputs)
        for b, enc_features in zip(self.blocks, encoder_features[::-1]):
            outputs = b(outputs, enc_features)
        outputs = self.last_conv(outputs)
        return outputs


class UNet(nn.Module):
    def __init__(self, img_size, num_classes, num_blocks=4, num_inputs=3):
        super().__init__()
        self.encoder = UNetEncoder(num_inputs, num_blocks)
        self.decoder = UNetDecoder(num_classes, num_blocks, num_classes)

    def forward(self, inputs):
        # inputs is B, 3, H, W

        # features is a list of features to
        # be taken as inputs, at different steps
        # by the decoder
        encoder_outputs, encoder_features = self.encoder(inputs)
        # encoder outputs is B, 32*(num_blocks+1), H/2^num_blocks, W/2^num_blocks
        # encoder_features is a list of num_blocks tensors
        outputs = self.decoder(encoder_outputs, encoder_features)
        # outputs is B, num_classes, H, W
        return outputs


def build_model(model_name, img_size, num_classes):
    if model_name not in available_models:
        raise RuntimeError(f"Unavailable model {model_name}")
    exec(f"m = {model_name}(img_size, num_classes)")
    return locals()["m"]


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    license = """
    models.py  Copyright (C) 2022  Jeremy Fix
    This program comes with ABSOLUTELY NO WARRANTY;
    This is free software, and you are welcome to redistribute it
    under certain conditions;
    """
    logging.info(license)
    mname = ["UNet"]

    for n in mname:
        m = build_model(n, (256, 256), 10)
        out = m(torch.zeros(2, 3, 256, 256))
        print(out.shape)
