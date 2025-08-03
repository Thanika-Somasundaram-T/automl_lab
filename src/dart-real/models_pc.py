import torch
import torch.nn as nn
import torch.nn.functional as F
import mup
from . import ops_pc as o

class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, stride, k=4):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in o.OPS:
            self._ops.append(o.PC_OPS(C_in, C_out, stride, primitive, k))
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, k=4):
        super().__init__()
        self.reduction = reduction
        self.steps = steps
        self.multiplier = multiplier

        # Preprocess inputs
        if reduction_prev:
            self.preprocess0 = o.OPS['factorized_reduce'](C_prev_prev, C, stride=2, affine=True)
        else:
            self.preprocess0 = o.OPS['conv_1x1'](C_prev_prev, C, stride=1, affine=True)
        self.preprocess1 = o.OPS['conv_1x1'](C_prev, C, stride=1, affine=True)

        # Build operations for the cell
        self._ops = nn.ModuleList()
        self._indices = []
        for i in range(self.steps):
            for j in range(2 + i):  # number of inputs to this node
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, C, stride, k)
                self._ops.append(op)
                self._indices.append(j)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self.steps):
            s = sum(self._ops[offset + j](states[self._indices[offset + j]], weights[offset + j]) for j in range(2 + i))
            offset += 2 + i
            states.append(s)
        return torch.cat(states[-self.multiplier:], dim=1)

class MultiHeadNetwork(nn.Module):
    def __init__(self, C, num_classes_dict, layers, criterion, k=4, base_width=16):
        super().__init__()
        self._C = C
        self._layers = layers
        self._steps = 4
        self._multiplier = 4
        self._criterion = criterion
        self.k = k
        self.base_width = base_width

        # Shared stem
        self.stem = o.MuConv2d(3, C * 3, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(C * 3)
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

        # Initial channels
        C_prev_prev = C * 3
        C_prev = C * 3
        C_curr = C
        reduction_prev = False

        # Build cells
        self.cells = nn.ModuleList()
        for i in range(layers):
            reduction = True if i in [layers // 3, 2 * layers // 3] else False
            cell = Cell(self._steps, self._multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, k)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, self._multiplier * C_curr
            if reduction:
                C_curr *= 2

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifiers = nn.ModuleDict({
            dataset_name: mup.MuReadout(C_prev, num_classes, readout_zero_init=True)
            for dataset_name, num_classes in num_classes_dict.items()
        })

        # Architecture parameters
        num_edges = sum(2 + i for i in range(self._steps))
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(num_edges, len(o.OPS)))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(num_edges, len(o.OPS)))
        self.betas_normal = nn.Parameter(1e-3 * torch.randn(num_edges))
        self.betas_reduce = nn.Parameter(1e-3 * torch.randn(num_edges))

    def forward_features(self, x):
        s0 = s1 = self.bn(self.stem(x))
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights_alpha = F.softmax(self.alphas_reduce, dim=-1)
                weights_beta = F.softmax(self.betas_reduce, dim=-1)
            else:
                weights_alpha = F.softmax(self.alphas_normal, dim=-1)
                weights_beta = F.softmax(self.betas_normal, dim=-1)
            weights = weights_alpha * weights_beta.unsqueeze(-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        return out.view(out.size(0), -1)

    def forward(self, x, dataset_name):
        features = self.forward_features(x)
        return self.classifiers[dataset_name](features)
