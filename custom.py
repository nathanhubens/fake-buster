from fastai.vision.all import *
from torch.hub import load_state_dict_from_url
from typing import Union, Tuple, Optional

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class Involution(nn.Module):
    """
    Implementation of `Involution: Inverting the Inherence of Convolution for Visual Recognition`.
    """
    def __init__(self, in_channels, out_channels, groups=4, kernel_size=7, stride=1, reduction_ratio=4):

        super().__init__()

        channels_reduced = max(1, in_channels // reduction_ratio)
        padding = kernel_size // 2

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, channels_reduced, 1),
            nn.BatchNorm2d(channels_reduced),
            nn.ReLU(inplace=True))

        self.span = nn.Conv2d(channels_reduced, kernel_size * kernel_size * groups, 1)
        self.unfold = nn.Unfold(kernel_size, padding=padding, stride=stride)
        
        self.resampling = None if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

    def forward(self, input_tensor):
        """
        Calculate Involution.
        override function from PyTorch.
        """
        _, _, height, width = input_tensor.size()
        if self.stride > 1:
            out_size = lambda x: (x + 2 * self.padding - self.kernel_size) // self.stride + 1
            height, width = out_size(height), out_size(width)
        uf_x = rearrange(self.unfold(input_tensor), 'b (g d k j) (h w) -> b g d (k j) h w',
                         g=self.groups, k=self.kernel_size, j=self.kernel_size, h=height, w=width)

        if self.stride > 1:
            input_tensor = F.adaptive_avg_pool2d(input_tensor, (height, width))
        kernel = rearrange(self.span(self.reduce(input_tensor)), 'b (k j g) h w -> b g (k j) h w',
                           k=self.kernel_size, j=self.kernel_size)

        out = rearrange(torch.einsum('bgdxhw, bgxhw -> bgdhw', uf_x, kernel), 'b g d h w -> b (g d) h w')
        
        if self.resampling:
            out = self.resampling(out)
            
        return out.contiguous()


class Involution2d(nn.Module):
    """
    This class implements the 2d involution proposed in:
    https://arxiv.org/pdf/2103.06255.pdf
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 sigma_mapping: Optional[nn.Module] = None,
                 kernel_size: Union[int, Tuple[int, int]] = (7, 7),
                 stride: Union[int, Tuple[int, int]] = (1, 1),
                 groups: int = 1,
                 reduce_ratio: int = 1,
                 dilation: Union[int, Tuple[int, int]] = (1, 1),
                 padding: Union[int, Tuple[int, int]] = (3, 3),
                 bias: bool = False,
                 **kwargs) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param sigma_mapping: (nn.Module) Non-linear mapping as introduced in the paper. If none BN + ReLU is utilized
        :param kernel_size: (Union[int, Tuple[int, int]]) Kernel size to be used
        :param stride: (Union[int, Tuple[int, int]]) Stride factor to be utilized
        :param groups: (int) Number of groups to be employed
        :param reduce_ratio: (int) Reduce ration of involution channels
        :param dilation: (Union[int, Tuple[int, int]]) Dilation in unfold to be employed
        :param padding: (Union[int, Tuple[int, int]]) Padding to be used in unfold operation
        :param bias: (bool) If true bias is utilized in each convolution layer
        :param **kwargs: Unused additional key word arguments
        """
        # Call super constructor
        super(Involution2d, self).__init__()
        # Check parameters
        assert isinstance(in_channels, int) and in_channels > 0, "in channels must be a positive integer."
        assert in_channels % groups == 0, "out_channels must be divisible by groups"
        assert isinstance(out_channels, int) and out_channels > 0, "out channels must be a positive integer."
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        assert isinstance(sigma_mapping, nn.Module) or sigma_mapping is None, \
            "Sigma mapping must be an nn.Module or None to utilize the default mapping (BN + ReLU)."
        assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple), \
            "kernel size must be an int or a tuple of ints."
        assert isinstance(stride, int) or isinstance(stride, tuple), \
            "stride must be an int or a tuple of ints."
        assert isinstance(groups, int), "groups must be a positive integer."
        assert isinstance(reduce_ratio, int) and reduce_ratio > 0, "reduce ratio must be a positive integer."
        assert isinstance(dilation, int) or isinstance(dilation, tuple), \
            "dilation must be an int or a tuple of ints."
        assert isinstance(padding, int) or isinstance(padding, tuple), \
            "padding must be an int or a tuple of ints."
        assert isinstance(bias, bool), "bias must be a bool"
        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.groups = groups
        self.reduce_ratio = reduce_ratio
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        #self.bias = bias
        # Init modules
        self.sigma_mapping = sigma_mapping if sigma_mapping is not None else nn.Sequential(
            nn.BatchNorm2d(num_features=self.out_channels // self.reduce_ratio, momentum=0.3), nn.ReLU())
        self.initial_mapping = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                         kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                         bias=bias) if self.in_channels != self.out_channels else nn.Identity()
        self.o_mapping = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)
        self.reduce_mapping = nn.Conv2d(in_channels=self.in_channels,
                                        out_channels=self.out_channels // self.reduce_ratio, kernel_size=(1, 1),
                                        stride=(1, 1), padding=(0, 0), bias=bias)
        self.span_mapping = nn.Conv2d(in_channels=self.out_channels // self.reduce_ratio,
                                      out_channels=self.kernel_size[0] * self.kernel_size[1] * self.groups,
                                      kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias)
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=dilation, padding=padding, stride=stride)

    def __repr__(self) -> str:
        """
        Method returns information about the module
        :return: (str) Info string
        """
        return ("{}({}, {}, kernel_size=({}, {}), stride=({}, {}), padding=({}, {}), "
                "groups={}, reduce_ratio={}, dilation=({}, {}), sigma_mapping={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.groups,
            self.reduce_mapping,
            self.dilation[0],
            self.dilation[1],
            #self.bias,
            str(self.sigma_mapping)
        ))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, out channels, height, width] (w/ same padding)
        """
        # Check input dimension of input tensor
        assert input.ndimension() == 4, \
            "Input tensor to involution must be 4d but {}d tensor is given".format(input.ndimension())
        # Save input shape
        batch_size, in_channels, height, width = input.shape
        # Unfold and reshape input tensor
        input_unfolded = self.unfold(self.initial_mapping(input))
        input_unfolded = input_unfolded.view(batch_size, self.groups, self.out_channels // self.groups,
                                             self.kernel_size[0] * self.kernel_size[1], height, width)
        # Generate kernel
        kernel = self.span_mapping(self.sigma_mapping(self.reduce_mapping(self.o_mapping(input))))
        kernel = kernel.view(
            batch_size, self.groups, self.kernel_size[0] * self.kernel_size[1], height, width).unsqueeze(dim=2)
        # Apply kernel to produce output
        output = (kernel * input_unfolded).sum(dim=3).view(batch_size, -1, height, width)
        return output


class XResNet(nn.Sequential):
    @delegates(ResBlock)
    def __init__(self, block, expansion, layers, p=0.0, c_in=3, n_out=1000, stem_szs=(32,32, 64),
                 widen=1.0, sa=False, act_cls=defaults.activation, ndim=2, ks=3, stride=2, **kwargs):
        store_attr('block,expansion,act_cls,ndim,ks')
        if ks % 2 == 0: raise Exception('kernel size has to be odd!')
        stem_szs = [c_in, *stem_szs]
        stem = [ConvLayer(stem_szs[i], stem_szs[i+1], ks=ks, stride=stride if i==0 else 1, act_cls=act_cls, ndim=ndim) for i in range(3)]
        #stem = [ConvLayer(stem_szs[i], stem_szs[i+1], ks=ks, stride=stride if i==0 else 1, act_cls=act_cls, ndim=ndim) for i in range(1)]

        block_szs = [int(o*widen) for o in [64,128,256,512] +[256]*(len(layers)-4)]
        #block_szs = [int(o*widen) for o in [64,128,256] +[256]*(len(layers)-3)]
        #print(block_szs)
        block_szs = [64//expansion] + block_szs
        #block_szs = [64] + block_szs
        #print(block_szs)

        blocks    = self._make_blocks(layers, block_szs, sa, stride, **kwargs)

        super().__init__(
            *stem, MaxPool(ks=ks, stride=stride, padding=ks//2, ndim=ndim),
            *blocks,
            AdaptiveAvgPool(sz=1, ndim=ndim), Flatten(), nn.Dropout(p),
            nn.Linear(block_szs[-1]*expansion, n_out),
        #    nn.Linear(block_szs[-1], n_out),
        )
        init_cnn(self)

    def _make_blocks(self, layers, block_szs, sa, stride, **kwargs):
        return [self._make_layer(ni=block_szs[i], nf=block_szs[i+1], blocks=l,
                                 stride=1 if i==0 else stride, sa=sa and i==len(layers)-4, **kwargs)
                for i,l in enumerate(layers)]

    def _make_layer(self, ni, nf, blocks, stride, sa, **kwargs):
        return nn.Sequential(
            *[self.block(self.expansion, ni if i==0 else nf, nf, stride=stride if i==0 else 1,
                      sa=sa and i==(blocks-1), act_cls=self.act_cls, ndim=self.ndim, ks=self.ks, **kwargs)
              for i in range(blocks)])


class XResNet_smol(nn.Sequential):
    @delegates(ResBlock)
    def __init__(self, block, expansion, layers, p=0.0, c_in=3, n_out=1000, stem_szs=32,
                 widen=1.0, sa=False, act_cls=defaults.activation, ndim=2, ks=3, stride=2, **kwargs):
        store_attr('block,expansion,act_cls,ndim,ks')
        if ks % 2 == 0: raise Exception('kernel size has to be odd!')
        stem_szs = [c_in, 32]
        stem = ConvLayer(stem_szs[0], stem_szs[1], ks=ks, stride=stride, act_cls=act_cls, ndim=ndim)
        #stem = [ConvLayer(stem_szs[i], stem_szs[i+1], ks=ks, stride=stride if i==0 else 1, act_cls=act_cls, ndim=ndim) for i in range(1)]

        block_szs = [int(o*widen) for o in [16, 32,64,128] +[256]*(len(layers)-3)]
        print(block_szs)
        print(expansion)
        #block_szs = [64//expansion] + block_szs
        print(block_szs)

        blocks    = self._make_blocks(layers, block_szs, sa, stride, **kwargs)

        super().__init__(
            *stem, MaxPool(ks=ks, stride=stride, padding=ks//2, ndim=ndim),
            *blocks,
            AdaptiveAvgPool(sz=1, ndim=ndim), Flatten(), nn.Dropout(p),
            nn.Linear(block_szs[-1]*expansion, n_out),
        #    nn.Linear(block_szs[-1], n_out),
        )
        init_cnn(self)

    def _make_blocks(self, layers, block_szs, sa, stride, **kwargs):
        return [self._make_layer(ni=block_szs[i], nf=block_szs[i+1], blocks=l,
                                 stride=1 if i==0 else stride, sa=sa and i==len(layers)-3, **kwargs)
                for i,l in enumerate(layers)]

    def _make_layer(self, ni, nf, blocks, stride, sa, **kwargs):
        return nn.Sequential(
            *[self.block(self.expansion, ni if i==0 else nf, nf, stride=stride if i==0 else 1,
                      sa=sa and i==(blocks-1), act_cls=self.act_cls, ndim=self.ndim, ks=self.ks, **kwargs)
              for i in range(blocks)])



class InvResBlock(Module):
    "Resnet block from `ni` to `nh` with `stride`"
    @delegates(ConvLayer.__init__)
    def __init__(self, expansion, ni, nf, stride=1, groups=1, reduction=None, nh1=None, nh2=None, dw=False, g2=1,
                 sa=False, sym=False, norm_type=NormType.Batch, act_cls=defaults.activation, ndim=2, ks=3,
                 pool=AvgPool, pool_first=True, **kwargs):
        norm2 = (NormType.BatchZero if norm_type==NormType.Batch else
                 NormType.InstanceZero if norm_type==NormType.Instance else norm_type)
        if nh2 is None: nh2 = nf
        if nh1 is None: nh1 = nh2
        nf,ni = nf*expansion,ni*expansion
        k0 = dict(norm_type=norm_type, act_cls=act_cls, ndim=ndim, **kwargs)
        k1 = dict(norm_type=norm2, act_cls=None, ndim=ndim, **kwargs)
        
        
        #convpath  = [Involution2d(ni,  nh2),
        #             Involution2d(nh2,  nf)]
        convpath  = [ConvLayer(ni,  nh1, 1, **k0),
                    Involution(nh1,  nh2),
                    ConvLayer(nh2,  nf, 1 , groups=g2, **k1)]
        #convpath = [Involution2d(ni, nf, reduce_ratio=4)]
        
        if reduction: convpath.append(SEModule(nf, reduction=reduction, act_cls=act_cls))
        if sa: convpath.append(SimpleSelfAttention(nf,ks=1,sym=sym))
        self.convpath = nn.Sequential(*convpath)
        idpath = []
        if ni!=nf: idpath.append(ConvLayer(ni, nf, 1, act_cls=None, ndim=ndim, **kwargs))
        self.idpath = nn.Sequential(*idpath)
        self.act = defaults.activation(inplace=True) if act_cls is defaults.activation else act_cls()

    def forward(self, x): 
        return self.act(self.convpath(x) + self.idpath(x))


class MBResBlock(Module):
    "Resnet block from `ni` to `nh` with `stride`"
    @delegates(ConvLayer.__init__)
    def __init__(self, expansion, ni, nf, stride=1, groups=1, reduction=None, nh1=None, nh2=None, dw=False, g2=1,
                 sa=False, sym=False, norm_type=NormType.Batch, act_cls=defaults.activation, ndim=2, ks=3,
                 pool=AvgPool, pool_first=True, **kwargs):
        norm2 = (NormType.BatchZero if norm_type==NormType.Batch else
                 NormType.InstanceZero if norm_type==NormType.Instance else norm_type)
        if nh2 is None: nh2 = nf
        if nh1 is None: nh1 = nh2
        nf,ni = nf*expansion,ni*expansion
        #nh1, nh2 = nh1*expansion, nh2*expansion

        k0 = dict(norm_type=norm_type, act_cls=act_cls, ndim=ndim, **kwargs)
        k1 = dict(norm_type=norm2, act_cls=None, ndim=ndim, **kwargs)
        
        
        convpath  = [ConvLayer(ni,  nh1, 1, **k0),
                     ConvLayer(nh1,  nh2, ks, stride=stride, groups=nh1 if dw else groups, **k0),
                     ConvLayer(nh2,  nf, 1 , groups=g2, **k1)
        ] if expansion == 1 else [
                     ConvLayer(ni,  nh1, 1, **k0),
                     ConvLayer(nh1, nh2, ks, stride=stride, groups=nh1 if dw else groups, **k0),
                     ConvLayer(nh2,  nf, 1, groups=g2, **k1)]
        
        
        if reduction: convpath.append(SEModule(nf, reduction=reduction, act_cls=act_cls))
        if sa: convpath.append(SimpleSelfAttention(nf,ks=1,sym=sym))
        self.convpath = nn.Sequential(*convpath)
        idpath = []
        if ni!=nf: idpath.append(ConvLayer(ni, nf, 1, act_cls=None, ndim=ndim, **kwargs))
        if stride!=1: idpath.insert((1,0)[pool_first], pool(stride, ndim=ndim, ceil_mode=True))
        self.idpath = nn.Sequential(*idpath)
        self.act = defaults.activation(inplace=True) if act_cls is defaults.activation else act_cls()

    def forward(self, x): return self.act(self.convpath(x) + self.idpath(x))


def _dw_xresnet(pretrained, expansion, layers, **kwargs):
    # TODO pretrain all sizes. Currently will fail with non-xrn50
    url = 'https://s3.amazonaws.com/fast-ai-modelzoo/xrn50_940.pth'
    res = XResNet(MBResBlock, expansion, layers, **kwargs)
    if pretrained: res.load_state_dict(load_state_dict_from_url(url, map_location='cpu')['model'], strict=False)
    return res


def _dw_xresnet_smol(pretrained, expansion, layers, **kwargs):
    # TODO pretrain all sizes. Currently will fail with non-xrn50
    url = 'https://s3.amazonaws.com/fast-ai-modelzoo/xrn50_940.pth'
    res = XResNet_smol(MBResBlock, expansion, layers, **kwargs)
    if pretrained: res.load_state_dict(load_state_dict_from_url(url, map_location='cpu')['model'], strict=False)
    return res

def _inv_xresnet(pretrained, expansion, layers, **kwargs):
    # TODO pretrain all sizes. Currently will fail with non-xrn50
    url = 'https://s3.amazonaws.com/fast-ai-modelzoo/xrn50_940.pth'
    res = XResNet(InvResBlock, expansion, layers, **kwargs)
    if pretrained: res.load_state_dict(load_state_dict_from_url(url, map_location='cpu')['model'], strict=False)
    return res

def _xresnet(pretrained, expansion, layers, **kwargs):
    # TODO pretrain all sizes. Currently will fail with non-xrn50
    url = 'https://s3.amazonaws.com/fast-ai-modelzoo/xrn50_940.pth'
    res = XResNet(ResBlock, expansion, layers, **kwargs)
    if pretrained: res.load_state_dict(load_state_dict_from_url(url, map_location='cpu')['model'], strict=False)
    return res


def dw_xresnet18(pretrained=False, **kwargs): return _dw_xresnet(pretrained, 1, [2, 2,  2, 2], n_out=2, dw=True, sa=True, reduction=16, **kwargs)
def dw_xresnet18_exp(pretrained=False, **kwargs): return _dw_xresnet_smol(pretrained, 2, [2, 2, 2, 2], n_out=2, dw=True, sa=True, reduction=16, **kwargs)
def dw_xresnet50(pretrained=False, **kwargs): return _dw_xresnet(pretrained, 4, [3, 4,  6, 3], n_out=2, dw=True, sa=True, reduction=16, **kwargs)
def inv_xresnet18(pretrained=False, **kwargs): return _inv_xresnet(pretrained, 2, [2, 2,  2, 2], n_out=2, dw=True, sa=True, reduction=16, **kwargs)
def xresnet18(pretrained=False, **kwargs): return _xresnet(pretrained, 2, [2, 2,  2, 2], n_out=2, **kwargs)