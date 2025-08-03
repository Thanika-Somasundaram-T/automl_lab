import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from . import utils
import mup

device = utils.get_device()

class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.C = C
        self.stride = stride
        
        for primitive in [
            'max_pool_3x3',
            'avg_pool_3x3',
            'skip_connect',
            'sep_conv_3x3',
            'sep_conv_5x5',
            'dil_conv_3x3',
            'dil_conv_5x5',
            'none'
        ]:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
}

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_prev_prev, C, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(C, affine=False)
            )
        self.preprocess1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_prev, C, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C, affine=False)
        )
        
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)

class PC_DARTSNetwork(nn.Module):
    def __init__(self, C, num_classes_dict, layers=8, criterion=nn.CrossEntropyLoss(), steps=4, multiplier=4, stem_multiplier=3):
        super(PC_DARTSNetwork, self).__init__()
        self._C = C
        self._num_classes_dict = num_classes_dict
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
 
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        
        # Create separate classifiers for each dataset
        self.classifiers = nn.ModuleDict()
        for name, num_classes in num_classes_dict.items():
            self.classifiers[name] = nn.Linear(C_prev, num_classes)
        
        self._init_mup()
        
        # Architecture parameters
        k = sum(1 for i in range(steps) for _ in range(2 + i))
        num_ops = len(OPS)
        
        self.alphas = nn.ParameterList()
        self.betas = nn.ParameterList()
        
        for i in range(k):
            self.alphas.append(nn.Parameter(1e-3 * torch.randn(num_ops)))
            self.betas.append(nn.Parameter(1e-3 * torch.randn(layers * 2)))
            
    def _init_mup(self):
        """Initialize layers with μP-aware scaling"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                # Hidden layers: input-dim scaling (fan_in)
                nn.init.kaiming_normal_(param, mode='fan_in')
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, input, dataset_name):
        s0 = s1 = self.stem(input)
        
        # Get architecture parameters
        alphas = [F.softmax(alpha, dim=-1) for alpha in self.alphas]
        betas = [F.softmax(beta, dim=-1) for beta in self.betas]
        
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = [alphas[k] for k in range(len(alphas))]
            else:
                weights = [alphas[k] for k in range(len(alphas))]
            
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        
        # Use the appropriate classifier based on dataset_name
        if dataset_name not in self.classifiers:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        return self.classifiers[dataset_name](out)