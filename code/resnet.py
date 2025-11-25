from functools import partial
from typing import Any, Callable, Optional, Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

# Hyperbolic layers import
from layers import HyperbolicConv2d, HyperLayer, HyperbolicActivation, HyperbolicAlgebra, HyperbolicRegression, HyperbolicEmbedding

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return HyperbolicConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return HyperbolicConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class HBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        input_shape: Optional[Tuple[int, int, int]] = None # Hyperbolic layers for now require explicit input shape handling
    ) -> None:
        super().__init__()
        self.m = HyperbolicAlgebra()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if input_shape is None:
            raise NotImplementedError("Hyperbolic blocks require explicit shape handling.")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        W = input_shape[1]
        H = input_shape[2]
        self.conv1 = conv3x3(inplanes, planes, stride) # (C, W, H) -> (planes, W/stride, H/stride)
        self.bn1 = HyperLayer(norm_layer, (planes, int(W/stride), int(H/stride)), planes)
        self.relu = HyperbolicActivation(nn.ReLU)
        self.conv2 = conv3x3(planes, planes) # (planes, W/stride, H/stride) -> (planes, W/stride, H/stride)
        self.bn2 = HyperLayer(norm_layer, (planes, int(W/stride), int(H/stride)), planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        self.m.group_mul(out, identity)
        out = self.relu(out)

        return out


class HBottleneck(BasicBlock):
    def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1, base_width = 64, dilation = 1, norm_layer = None):
        raise NotImplementedError("Hyperbolic bottleneck not yet implemented")


class HResNet(nn.Module):
    def __init__(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        input_shape: Optional[Tuple[int,int,int]] = (3, 224, 224)
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        W = input_shape[1]
        H = input_shape[2]
        self.groups = groups
        self.base_width = width_per_group

        self.embd = HyperbolicEmbedding()
        self.conv1 = HyperbolicConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # (C, W, H) -> (self.inplanes, W/2, H/2)
        self.bn1 = HyperLayer(norm_layer, (self.inplanes, int(W/2), int(H/2)), self.inplanes)
        self.relu = HyperbolicActivation(nn.ReLU)
        self.maxpool = HyperLayer(nn.MaxPool2d, (self.inplanes, int(W/2), int(H/2)), kernel_size=3, stride=2, padding=1)
        # (self.inplanes, W/2, H/2) -> (self.inplanes, W/4, H/4)
        self.layer1 = self._make_layer(
            block, 64, layers[0],
            input_shape=(self.inplanes, int(W/4), int(H/4))
            )
        # (self.inplanes, W/4, H/4) -> (64, W/4, H/4) since no stride
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
            input_shape=(64, int(W/4), int(H/4))
            )
        # (64, W/4, H/4) -> (128, W/(2**(2+layers[1])), H/(2**(2+layers[1]))) since stride 2
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
            input_shape=(128, int(W/8), int(H/8))
            )
        # (128, W/(2**(2+layers[1])), H/(2**(2+layers[1]))) -> (256, W/(2**(2+layers[1] + layers[2])),  H/(2**(2+layers[1] + layers[2])) ) since stride 2
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
            input_shape=(256, int(W/16), int(H/16))
            )
        # (256, W/(2**(2+layers[1] + layers[2])),  H/(2**(2+layers[1] + layers[2]))) -> (512, W/(2**(2+layers[1] + layers[2] + layers[3])),  H/(2**(2+layers[1] + layers[2] + layers[3])))        

        self.avgpool = HyperLayer(nn.AdaptiveAvgPool2d,
                                  (512, int(W/32),  int(H/32)),
                                  (1, 1))
        self.fc = HyperbolicRegression((512 * block.expansion)+1, num_classes)
        self.fc.bias.register_hook(lambda x: torch.clip(x, -10, 10))

        for m in self.modules():
            if isinstance(m, HyperbolicConv2d):
                nn.init.kaiming_normal_(m.fiberConv2d.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, HyperLayer) and isinstance(m.layer, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.layer.weight, 1)
                nn.init.constant_(m.layer.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, HBottleneck) and m.bn3.layer.weight is not None:
                    nn.init.constant_(m.bn3.layer.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, HBasicBlock) and m.bn2.layer.weight is not None:
                    nn.init.constant_(m.bn2.layer.weight, 0)  # type: ignore[arg-type]


    def _make_layer(
        self,
        block: type[Union[HBasicBlock, HBottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        input_shape: Optional[Tuple[int,int,int]] = None
    ) -> nn.Sequential:
        if input_shape is None:
            raise NotImplementedError("Hyperbolic layers require explicit shape handling")
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                HyperLayer(norm_layer,
                           (planes * block.expansion, int(input_shape[1]/stride), int(input_shape[2]/stride)),
                           (planes * block.expansion)),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, input_shape
            ) # only the first block is strided
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups, # this is always one in the original implementation
                    base_width=self.base_width, # also constant in the original implementation
                    dilation=self.dilation, # we ignore this for now?
                    norm_layer=norm_layer, # usual bnorm, does not change shape
                    input_shape=(input_shape[0], int(input_shape[1]/stride), int(input_shape[2]/stride))
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.embd(x.flatten(start_dim=1))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)



        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)