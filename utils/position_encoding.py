import torch
from torch import nn
import math
import numpy as np
from utils.misc import NestedTensor

"""
1D position encoding and 2D postion encoding
The code is modified based on DETR of Facebook: 
https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
"""

class PositionEmbeddding1D(nn.Module):
    """
    1D position encoding
    Based on Attetion is all you need paper and DETR PositionEmbeddingSine class
    """
    def __init__(self, temperature = 10000, normalize = False, scale = None):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, L: int, N: int, E: int):
        """
        Args:
            L for length, N for batch size, E for embedding size (dimension of transformer).

        Returns:
            pos: position encoding, with shape [L, N, E]
        """
        pos_embed = torch.ones(N, L, dtype = torch.float32).cumsum(axis = 1)
        dim_t = torch.arange(E, dtype = torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / E)
        if self.normalize:
            eps = 1e-6
            pos_embed = pos_embed / (L + eps) * self.scale

        pos_embed = pos_embed[:, :, None] / dim_t
        pos_embed = torch.stack((pos_embed[:, :, 0::2].sin(), pos_embed[:, :, 1::2].cos()), dim = 3).flatten(2)
        pos_embed = pos_embed.permute(1, 0, 2)
        pos_embed.requires_grad_(False)
        
        return pos_embed

class PositionEmbeddding2D(nn.Module):
    """
    2D position encoding, borrowed from DETR PositionEmbeddingSine class
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """
    def __init__(self, temperature=10000, normalize=False, scale=None, device = torch.device('cuda:0')):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.device = device
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, N: int, E: int, H: int, W: int):
        """
        Args:
            N for batch size, E for embedding size (channel of feature), H for height, W for width

        Returns:
            pos_embed: positional encoding with shape (N, E, H, W)
        """
        assert E % 2 == 0, "Embedding size should be even number"

        y_embed = torch.ones(N, H, W, dtype=torch.float32, device = self.device).cumsum(dim = 1)
        x_embed = torch.ones(N, H, W, dtype=torch.float32, device = self.device).cumsum(dim = 2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(E//2, dtype=torch.float32, device=self.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (E//2))

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos_embed.requires_grad_(False)
        return pos_embed


class PositionEmbeddding3D(nn.Module):
    """
    3D position encoding
    """
    def __init__(self, E, T, temperature=10000, normalize=False, scale=None, device = torch.device('cuda:0')):
        """
        E: embedding size, i.e. pos feature length
        T: video clip length
        """
        super().__init__()
        self.E = E
        self.T = T
        self.temperature = temperature
        self.normalize = normalize
        self.device = device
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    
    def forward(self, tensorlist: NestedTensor):
        """
        Args:
            tensorlist: NestedTensor which includes feature maps X and corresponding mask
            X: tensor with shape (N*T, C, H, W), N for batch size, C for channel of features, T for time, H for height, W for width
            mask: None or tensor with shape (N*T, H, W)
        Returns:
            pos_embed: positional encoding with shape (N, E, T, H, W)
        """
        NT, C, H, W= tensorlist.tensors.shape
        N = NT//self.T
        mask = tensorlist.mask
        assert self.E % 3 == 0, "Embedding size should be divisible by 3"

        if mask is None:
            t_embed = torch.ones(N, self.T, H, W, dtype = torch.float32, device = self.device).cumsum(dim = 1)
            y_embed = torch.ones(N, self.T, H, W, dtype=torch.float32, device = self.device).cumsum(dim = 2)
            x_embed = torch.ones(N, self.T, H, W, dtype=torch.float32, device = self.device).cumsum(dim = 3)
        else:
            mask = mask.reshape(N, self.T, H, W)
            #binary mask, 1 for the image area, 0 for the padding area
            t_embed = mask.cumsum(dim = 1, dtype = torch.float32).to(self.device)
            y_embed = mask.cumsum(dim = 2, dtype = torch.float32).to(self.device)
            x_embed = mask.cumsum(dim = 3, dtype = torch.float32).to(self.device)
        if self.normalize:
            eps = 1e-6
            t_embed = t_embed / (t_embed[:, :-1, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale
        
        dim_t = torch.arange(self.E//3, dtype=torch.float32, device=self.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.E//3))

        pos_t = t_embed[:, :, :, :, None] / dim_t
        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t

        pos_t = torch.stack((pos_t[:, :, :, :, 0::2].sin(), pos_t[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)

        pos_embed = torch.cat((pos_t, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)
        pos_embed.requires_grad_(False)
        
        return pos_embed