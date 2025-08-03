import torch
import torch.nn as nn
import mup
from ops import OPS

class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in OPS:
            op = OPS[primitive](C, stride, affine=True)
            self._ops.append(op)

    def forward(self, x, weights):
        # weights is softmax over alphas for this op
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C, reduction, reduction_prev):
        super().__init__()
        self.reduction = reduction
        self.steps = steps
        self.multiplier = multiplier

        self.preprocess0 = OPS['conv_1x1'](C if reduction_prev else C, 1, affine=True)
        self.preprocess1 = OPS['conv_1x1'](C, 1, affine=True)

        self._ops = nn.ModuleList()
        for i in range(self.steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self.steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self.multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion):
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = 4
        self._multiplier = 4

        self.stem = mup.MuConv2d(3, C * 3, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(C * 3)

        C_prev_prev, C_prev, C_curr = C * 3, C * 3, C

        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            reduction = True if i in [layers // 3, 2 * layers // 3] else False
            cell = Cell(self._steps, self._multiplier, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, self._multiplier * C_curr
            if reduction:
                C_curr *= 2

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = mup.MuReadout(C_prev, num_classes)


        # Architecture parameters (alphas)
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(len(OPS), self._steps * (self._steps + 1) // 2))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(len(OPS), self._steps * (self._steps + 1) // 2))

        mup.make_max_param(self)

    def forward(self, input):
        s0 = s1 = self.bn(self.stem(input))
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = torch.softmax(self.alphas_reduce, dim=0)
            else:
                weights = torch.softmax(self.alphas_normal, dim=0)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)
