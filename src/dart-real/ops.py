import torch
import torch.nn as nn
import mup

# Base ops that DARTS uses, with μP conv and linear where applicable

class MuPConv3x3(mup.MuConv2d):
    def __init__(self, C_in, C_out, stride, affine=True):
        super().__init__(C_in, C_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        return self.bn(self.conv(x))


class MuPConv1x1(mup.MuConv2d):
    def __init__(self, C_in, C_out, stride, affine=True):
        super().__init__(C_in, C_out, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        return self.bn(self.conv(x))


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            MuPConv3x3(C_in, C_out, stride),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        # downsample by stride
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = mup.MuConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = mup.MuConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


# Define OPS dict for searching
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'conv_3x3': lambda C, stride, affine: MuPConv3x3(C, C, stride, affine=affine),
    'conv_1x1': lambda C, stride, affine: MuPConv1x1(C, C, stride, affine=affine),
    # add other ops as needed
}
