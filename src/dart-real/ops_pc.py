import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mup import MuReadout

# ---- μP Conv2d ----
class MuConv2d(nn.Conv2d):
    """μP-scaled Conv2d with proper initialization"""
    def __init__(self, *args, **kwargs):
        self._initialized = False   # <-- define first so reset_parameters can access it
        super().__init__(*args, **kwargs)
        self.reset_parameters()     # call again to use our custom init
    
    def reset_parameters(self):
        if not self._initialized:
            nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
            if self.bias is not None:
                fan_in = self.weight.shape[1] * self.kernel_size[0] * self.kernel_size[1]
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
            self._initialized = True

class MuPConv3x3(nn.Module):
    def __init__(self, C_in, C_out, stride, affine=True):
        super().__init__()
        self.conv = MuConv2d(C_in, C_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        if affine:
            nn.init.ones_(self.bn.weight)
            nn.init.zeros_(self.bn.bias)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class MuPConv1x1(nn.Module):
    def __init__(self, C_in, C_out, stride, affine=True):
        super().__init__()
        self.conv = MuConv2d(C_in, C_out, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        if affine:
            nn.init.ones_(self.bn.weight)
            nn.init.zeros_(self.bn.bias)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

# ---- Channel Shuffle ----
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batchsize, -1, height, width)

# ---- DARTS ops ----
class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class Identity(nn.Module):
    def forward(self, x):
        return x

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = MuConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = MuConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)

class SepConv(nn.Module):
    """Depthwise separable conv used in DARTS"""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.depthwise = MuConv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in, bias=False)
        self.pointwise = MuConv2d(C_in, C_out, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    def forward(self, x):
        x = self.relu(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.bn(x)

class DilConv(nn.Module):
    """Dilated conv used in DARTS"""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv = MuConv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        return self.bn(x)

# ---- Partial Channel ----
class PartialChannelOp(nn.Module):
    def __init__(self, op_fn, C_in, C_out, stride, k=4):
        super().__init__()
        self.k = k
        self._op = op_fn(C_in // k, C_out // k, stride, affine=False)
    def forward(self, x):
        C = x.shape[1]
        x1 = x[:, :C // self.k, :, :]
        x2 = x[:, C // self.k:, :, :]
        out = self._op(x1)
        if hasattr(self._op, 'conv') and self._op.conv.stride != (1, 1):
            x2 = F.avg_pool2d(x2, kernel_size=self._op.conv.stride, stride=self._op.conv.stride)
        out = torch.cat([out, x2], dim=1)
        return channel_shuffle(out, self.k)

def PC_OPS(C_in, C_out, stride, primitive, k=4):
    return PartialChannelOp(lambda Csub_in, Csub_out, s, affine: OPS[primitive](Csub_in, Csub_out, s, affine),
                            C_in, C_out, stride, k)

OPS = {
    'none': lambda C_in, C_out, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C_in, C_out, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C_in, C_out, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C_in, C_out, stride, affine: Identity() if stride == 1 else FactorizedReduce(C_in, C_out, affine=affine),
    'sep_conv_3x3': lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 5, stride, 2, affine=affine),
    'dil_conv_3x3': lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 3, stride, 2, dilation=2, affine=affine),
    'conv_1x1': lambda C_in, C_out, stride, affine: MuPConv1x1(C_in, C_out, stride, affine=affine),
    'factorized_reduce': lambda C_in, C_out, stride, affine: FactorizedReduce(C_in, C_out, affine=affine),
}
