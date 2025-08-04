from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


PRIMITIVES = [
            'max_pool_3x3',
            'avg_pool_3x3',
            'skip_connect',
            'sep_conv_3x3',
            'sep_conv_5x5',
            'dil_conv_3x3',
            'dil_conv_5x5',
            'none'
        ]
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.C = C
        self.stride = stride
        
        for primitive in PRIMITIVES:
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

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

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
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
            
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

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
            
class Network(nn.Module):
    def __init__(self, C, num_classes_dict, layers=8, criterion=nn.CrossEntropyLoss(), steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
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

        # Count how many normal and reduce cells
        self.normal_cell_indices = []
        self.reduce_cell_indices = []

        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                self.reduce_cell_indices.append(i)
            else:
                reduction = False
                self.normal_cell_indices.append(i)

            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        # Separate classifiers for each dataset
        self.classifiers = nn.ModuleDict()
        for name, num_classes in num_classes_dict.items():
            self.classifiers[name] = nn.Linear(C_prev, num_classes)
            
        # Architecture parameters per cell
        self._initialize_alphas()

    # def _init_mup(self):
    #     """Initialize layers with μP-aware scaling"""
    #     for name, param in self.named_parameters():
    #         if 'weight' in name and len(param.shape) >= 2:
    #             torch.nn.init.kaiming_normal_(param.data, mode='fan_in')
    #         elif 'bias' in name:
    #             nn.init.zeros_(param)
                
    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, dataset_name):
        s0 = s1 = self.stem(input)

        normal_cell_ptr = 0
        reduce_cell_ptr = 0

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                alphas = F.softmax(self.alphas_reduce[reduce_cell_ptr], dim=-1)
                reduce_cell_ptr += 1
            else:
                alphas = F.softmax(self.alphas_normal[normal_cell_ptr], dim=-1)
                normal_cell_ptr += 1

            weights = [alphas[j] for j in range(len(cell._ops))]

            s0, s1 = s1, cell(s0, s1, weights)

        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)

        if dataset_name not in self.classifiers:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return self.classifiers[dataset_name](out)
    
    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target) 
    
    
    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for _ in range(2 + i))
        num_ops = len(PRIMITIVES)

        # Create ParameterLists for alphas per cell type
        self.alphas_normal = nn.ParameterList([
            nn.Parameter(1e-3 * torch.randn(k, num_ops)) for _ in self.normal_cell_indices
        ])
        self.alphas_reduce = nn.ParameterList([
            nn.Parameter(1e-3 * torch.randn(k, num_ops)) for _ in self.reduce_cell_indices
        ])
        self._arch_parameters = [
        self.alphas_normal,
        self.alphas_reduce,
        ]
    
    def arch_parameters(self):
        return self._arch_parameters
    
    
    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        # gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_normal = []
        for alpha in self.alphas_normal:
            weights = F.softmax(alpha, dim=-1).data.cpu().numpy()
            gene_normal.append(_parse(weights))
        gene_normal = sum(gene_normal, [])  # flatten
        # gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
        gene_reduce = []
        for alpha in self.alphas_normal:
            weights = F.softmax(alpha, dim=-1).data.cpu().numpy()
            gene_reduce.append(_parse(weights))
        gene_reduce = sum(gene_reduce, [])  # flatten

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
    
    
class CellFixed(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, steps=4, multiplier=4):
        super(CellFixed, self).__init__()
        self.reduction = reduction
        self._steps = steps
        self._multiplier = multiplier

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, False)
            self._ops.append(op)
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i + 1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i + 1]
            s = op1(h1) + op2(h2)
            states.append(s)
        return torch.cat([states[i] for i in self._concat], dim=1)


class NetworkFixed(nn.Module):
    def __init__(self, C, num_classes_dict, layers, genotype, steps=4, multiplier=4, stem_multiplier=3):
        super(NetworkFixed, self).__init__()
        self._C = C
        self._num_classes_dict = num_classes_dict
        self._layers = layers
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
            cell = CellFixed(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, steps, multiplier)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.classifiers = nn.ModuleDict()
        for name, num_classes in num_classes_dict.items():
            self.classifiers[name] = nn.Linear(C_prev, num_classes)

    def forward(self, x, dataset_name):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        if dataset_name not in self.classifiers:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return self.classifiers[dataset_name](out)