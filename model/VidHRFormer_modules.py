import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import to_2tuple
import math
import torch.nn.functional as F
import copy
from utils.position_encoding import PositionEmbeddding1D
from .MultiHeadAttentionRPE import MultiheadAttentionRPE

"""
Encoder, same for auto-regressive and non-autoregressive models
"""
class VidHRFormerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, local_window_pos_embed, temporal_pos_embed):
        output = src
        for layer in self.layers:
            output = layer(output, local_window_pos_embed, temporal_pos_embed)
        if self.norm is not None:
            output = self.norm(output)

        return output

class VidHRFormerBlockEnc(nn.Module):
    def __init__(self, encH, encW, embed_dim, num_heads, window_size = 7, dropout = 0., drop_path = 0., Spatial_FFN_hidden_ratio = 4, dim_feedforward = 1024, far = False, rpe = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.dropout = dropout
        self.Spatial_FFN_hidden_ratio = Spatial_FFN_hidden_ratio

        self.SLMHSA = SpatialLocalMultiheadAttention(embed_dim, num_heads, window_size, dropout, rpe)
        MlpDWBN_layer_norm = False
        if far:
            MlpDWBN_layer_norm = True
        self.SpatialFFN = MlpDWBN(encH, encW, embed_dim, hidden_features = int(Spatial_FFN_hidden_ratio*embed_dim), out_features = embed_dim, drop = dropout, AR_model = MlpDWBN_layer_norm)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm3 = nn.LayerNorm(embed_dim)
        self.temporal_MHSA = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.activation = nn.GELU()
        self.drop1 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.drop2 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.drop3 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.norm4 = nn.LayerNorm(embed_dim)

        self.far = far

    def forward(self, x, local_window_pos_embed, temporal_pos_embed):
        """
        x: (N, T, H, W, C)
        local_window_pos_embed: (window_size, window_size, C)
        temporal_pos_embed: (T, C)
        Return: (N, T, H, W, C)
        """
        N, T, H, W, C = x.shape
        x = x + self.drop_path(self.SLMHSA(self.norm1(x), local_window_pos_embed)) #spatial local window self-attention, and skip connection
        
        #Conv feed-forward, different local window information interacts
        x = x + self.drop_path(self.SpatialFFN(self.norm2(x)))#(N, T, H, W, C)

        #temporal attention
        x = x.permute(1, 0, 2, 3, 4).reshape(T, N*H*W, C)
        x1 = self.norm3(x)
        if self.far:
            #create attention mask for temporal self-attention
            attn_mask = torch.triu(torch.ones(T, T), diagonal = 1) == 1
            x = x + self.drop1(self.temporal_MHSA(x1 + temporal_pos_embed[:, None, :], 
                               x1 + temporal_pos_embed[:, None, :], 
                               x1, 
                               attn_mask = attn_mask.to(x1.device))[0])
        else:
            x = x + self.drop1(self.temporal_MHSA(x1 + temporal_pos_embed[:, None, :], x1 + temporal_pos_embed[:, None, :], x1)[0])

        #output feed-forward
        x1 = self.norm4(x)
        x1 = self.linear2(self.drop2(self.activation(self.linear1(x1))))
        x = x + self.drop3(x1)

        x = x.reshape(T, N, H, W, C).permute(1, 0, 2, 3, 4)

        return x

####################################End of Transformer Encoder modules#################################
class VidHRformerDecoderNAR(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, query_pos, memory, local_window_pos_embed, future_query_temporal_pos_embed, TS_local_pos_embed, past_query_temporal_pos_embed):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, query_pos, memory, local_window_pos_embed, future_query_temporal_pos_embed, TS_local_pos_embed, past_query_temporal_pos_embed)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        
        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output

class VidHRFormerBlockDecNAR(nn.Module):
    def __init__(self, encH, encW, embed_dim, num_heads, window_size = 7, dropout = 0., drop_path = 0., Spatial_FFN_hidden_ratio = 4, dim_feedforward = 1024, TSLMA_flag = False, rpe = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.dropout = dropout
        self.Spatial_FFN_hidden_ratio = Spatial_FFN_hidden_ratio

        #self-attention of query, same as encoder
        self.SLMHSA = SpatialLocalMultiheadAttention(embed_dim, num_heads, window_size, dropout, rpe)
        self.SpatialFFN = MlpDWBN(encH, encW, embed_dim, hidden_features = int(Spatial_FFN_hidden_ratio*embed_dim), out_features = embed_dim, drop = dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm3 = nn.LayerNorm(embed_dim)
        self.temporal_MHSA = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        #Feed forward after termporal self-attention
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.activation = nn.GELU()
        self.drop2 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.drop3 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.norm4 = nn.LayerNorm(embed_dim)

        #encoder-decoder attention, follow with conv feed-forward
        self.TSLMA_flag = TSLMA_flag
        if TSLMA_flag:
            self.TSLMA = TemporalSpatialLocalMultiheadAttention(embed_dim, num_heads, window_size, dropout)
        else:
            self.EncDecAttn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=dropout)
        self.SpatialFFN1 = MlpDWBN(encH, encW, embed_dim, hidden_features = int(Spatial_FFN_hidden_ratio*embed_dim), out_features = embed_dim, drop = dropout)
        self.norm5 = nn.LayerNorm(embed_dim)
        self.norm6 = nn.LayerNorm(embed_dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, tgt, query_pos, memory, local_window_pos_embed, future_query_temporal_pos_embed, TS_local_pos_embed, past_query_temporal_pos_embed):
        """
        tgt: (N, T2, H, W, C)
        query_pos: (N, T2, H, W, C)
        memory: (N, T1, H, W, C)
        local_window_pos_embed: (window_size, window_size, C)
        query_temporal_pos_embed: (T2, C)
        TS_local_pos_embed: (T1+T2, window_size, window_size, C)

        Return: (N, T2, H, W, C)
        """
        N, T2, H, W, C = tgt.shape
        tgt2 = self.norm1(tgt)
        tgt2_query_pos = tgt2 + query_pos
        tgt2 = tgt + self.drop_path(self.SLMHSA(tgt2_query_pos, local_window_pos_embed, value = tgt2)) #spatial local window self-attention, and skip connection
        #Conv feed-forward, different local window information interacts
        tgt2 = tgt2 + self.drop_path(self.SpatialFFN(self.norm2(tgt2))) #(N, T, H, W, C)

        #query temporal self-attention
        tgt2 = tgt2.permute(1, 0, 2, 3, 4).reshape(T2, N*H*W, C)
        tgt = self.norm3(tgt2)
        tgt2 = tgt2 + self.drop1(self.temporal_MHSA(tgt + future_query_temporal_pos_embed[:, None, :], 
                                              tgt + future_query_temporal_pos_embed[:, None, :], 
                                              tgt)[0])
        
        #feed-forward after temporal self-attention
        tgt = self.norm4(tgt2)
        tgt = self.linear2(self.drop2(self.activation(self.linear1(tgt))))
        tgt2 = tgt2 + self.drop3(tgt)

        if self.TSLMA_flag:
            tgt2 = tgt2.reshape(T2, N, H, W, C).permute(1, 0, 2, 3, 4)
            #encoder-decoder attention (temporal local spatial full attention)
            tgt = self.norm5(tgt2)
            tgt2 = tgt2 + self.drop_path1(self.TSLMA(memory, tgt + query_pos, TS_local_pos_embed))
        else:
            tgt = self.norm5(tgt2)
            T1 = memory.shape[1]
            memory = memory.permute(1, 0, 2, 3, 4).reshape(T1, N*H*W, C)
            query_pos = query_pos.permute(1, 0, 2, 3, 4).reshape(T2, N*H*W, C)
            tgt2 = tgt2 + self.drop_path1(self.EncDecAttn(query = tgt+query_pos+future_query_temporal_pos_embed[:, None, :], 
                                                          key = memory+past_query_temporal_pos_embed[:, None, :], value = memory)[0])
            tgt2 = tgt2.reshape(T2, N, H, W, C).permute(1, 0, 2, 3, 4)

        #another Conv feed-forward, different local window information interacts
        tgt2 = tgt2 + self.drop_path1(self.SpatialFFN1(self.norm6(tgt2)))

        return tgt2


####################################End of non-autoregressive Transformer decoder###################################

"""
Modules for the VidHRFormer
"""
class TemporalSpatialLocalMultiheadAttention(nn.Module):
    """
    Modified based on https://github.com/HRNet/HRFormer/blob/main/cls/models/modules/multihead_isa_attention.py
    Based on SpatialLocalMultiheadAttention, but also attends along time dim, used for the encoder-decoder attention
    Args:
        embed_dim (int): Number of input channels.
        window_size (tuple[int]): Window size.
        num_heads (int): Number of attention heads.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        window_size=7,
        dropout = 0.
    ):
        super().__init__()

        self.dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.dropout = dropout

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.pad_helper = PadBlock(window_size)
        self.permute_helper = TemporalLocalPermuteModule(window_size)

    def forward(self, memory, query, TS_local_pos_embed):
        """
        memory: (N, T1, H, W, C)
        query: (N, T2, H, W, C)
        TS_local_pos_embed: (T1+T2, window_size, window_size, C)
        return:
           (N, T2, H, W, C)
        """
        # attention
        # pad
        N, T1, H, W, C = memory.shape
        memory = memory.view(N*T1, H, W, C)
        memory_pad = self.pad_helper.pad_if_needed(memory, memory.size())
        _, H_pad, W_pad, C = memory_pad.shape
        memory_pad = memory_pad.view(N, T1, H_pad, W_pad, C)

        N, T2, H, W, C = query.shape
        query = query.view(N*T2, H, W, C)
        query_pad = self.pad_helper.pad_if_needed(query, query.size())
        _, H_pad, W_pad, C = query_pad.shape
        query_pad = query_pad.view(N, T2, H_pad, W_pad, C)

        # permute
        memory_permute = self.permute_helper.permute(memory_pad, memory_pad.size()) #(T1*window_size*window_size, N*H/window_size*W/window_size, C)
        query_permute = self.permute_helper.permute(query_pad, query_pad.size()) #(T2*window_size*window_size, N*H/window_size*W/window_size, C)

        # attention
        out = self.attn(query = query_permute + TS_local_pos_embed[T1:T1+T2, ...].flatten(0, 2)[:, None, :],
                        key = memory_permute + TS_local_pos_embed[0:T1, ...].flatten(0, 2)[:, None, :],
                        value = memory_permute)[0]# (T2*window_size*window_size, N*H/window_size*W/window_size, C)
        # reverse permutation
        out = self.permute_helper.rev_permute(out, query_pad.size())
        out = out.reshape(N*T2, H_pad, W_pad, C) #(N*T2, H_pad, W_pad, C)

        # de-pad, pooling with `ceil_mode=True` will do implicit padding, so we need to remove it, too
        out = self.pad_helper.depad_if_needed(out, (N*T2, H, W, C))

        return out.view(N, T2, H, W, C)


class SpatialLocalMultiheadAttention(nn.Module):
    """
    Modified based on https://github.com/HRNet/HRFormer/blob/main/cls/models/modules/multihead_isa_attention.py
    local spatial window attention with absolute positional encoding, i.e. based the standard nn.MultiheadAttention module
    Args:
        embed_dim (int): Number of input channels.
        window_size (tuple[int]): Window size.
        num_heads (int): Number of attention heads.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        window_size=7,
        dropout = 0.,
        rpe = False
    ):
        super().__init__()

        self.dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.dropout = dropout

        self.rpe = rpe
        if rpe:
            self.attn = MultiheadAttentionRPE(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, 
                                              rpe = True, window_size = window_size)
        else:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.pad_helper = PadBlock(window_size)
        self.permute_helper = LocalPermuteModule(window_size)

    def forward(self, x, local_pos_embed, value = None):
        """
        x: (N, T, H, W, C)
        value: value should be None for encoder self-attention, value is not None for the Transformer decoder self-attention
        local_pos_embed: (window_size, window_size, C)
        return:
           (N, T, H, W, C)
        """
        N, T, H, W, C = x.shape
        x = x.view(N*T, H, W, C)

        # attention
        # pad
        x_pad = self.pad_helper.pad_if_needed(x, x.size())
        # permute
        x_permute = self.permute_helper.permute(x_pad, x_pad.size()) #(window_size*window_size, N*T*H/window_size*W/window_size, C)
        
        if self.rpe:
            k = q = x_permute
        else:
            k = q = x_permute + local_pos_embed.flatten(0, 1)[:, None, :]

        if value is not None:
            value = value.view(N*T, H, W, C)
            value_pad = self.pad_helper.pad_if_needed(value, value.size())
            value_permute = self.permute_helper.permute(value_pad, value_pad.size()) #(window_size*window_size, N*T*H/window_size*W/window_size, C)
            # attention
            out = self.attn(q, k, value = value_permute)[0]
        else:
            out = self.attn(q, k, value = x_permute)[0]
        # reverse permutation
        out = self.permute_helper.rev_permute(out, x_pad.size()) #(N*T, H, W, C)

        # de-pad, pooling with `ceil_mode=True` will do implicit padding, so we need to remove it, too
        out = self.pad_helper.depad_if_needed(out, x.size())

        return out.view(N, T, H, W, C)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, NT):
        # calculate flops for 1 window with token length of NT
        flops = 0
        # qkv = self.qkv(x)
        flops += NT * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * NT * (self.dim // self.num_heads) * NT
        #  x = (attn @ v)
        flops += self.num_heads * NT * NT * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += NT * self.dim * self.dim
        return flops


class MlpDWBN(nn.Module):
    """
    https://github.com/HRNet/HRFormer/blob/main/cls/models/modules/ffn_block.py
    """
    def __init__(
        self,
        encH,
        encW,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        dw_act_layer=nn.GELU,
        drop=0.0,
        AR_model = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act1 = act_layer()
        if AR_model:
            self.norm1 = nn.LayerNorm((hidden_features, encH, encW))
        else:
            self.norm1 = nn.BatchNorm2d(hidden_features)
        self.dw3x3 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            groups=hidden_features,
            padding=1,
        )
        self.act2 = dw_act_layer()
        if AR_model:
            self.norm2 = nn.LayerNorm((hidden_features, encH, encW))
        else:
            self.norm2 = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act3 = act_layer()
        if AR_model:
            self.norm3 = nn.LayerNorm((out_features, encH, encW))
        else:
            self.norm3 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)

        self.out_features = out_features

    def forward(self, x):
        """
        x: (N, T, H, W, C)
        """
        N, T, H, W, C = x.shape
        x = x.view(N*T, H, W, C).permute(0, 3, 1, 2)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dw3x3(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm3(x)
        x = self.act3(x)
        x = self.drop(x)

        return x.permute(0, 2, 3, 1).reshape(N, T, H, W, self.out_features)

class TemporalLocalPermuteModule(object):
    """ Permute the feature map to gather pixels in spatial local groups, and the reverse permutation
    https://github.com/HRNet/HRFormer/blob/main/cls/models/modules/multihead_isa_attention.py
    """
    def __init__(self, local_group_size=7):
        self.lgs = local_group_size
        if not isinstance(self.lgs, (tuple, list)):
            self.lgs = to_2tuple(self.lgs)
        assert len(self.lgs) == 2

    def permute(self, x, size):
        """
        x: (N, T, H, W, C)
        return: (T*local_group_size*local_group_size, N*H/local_group_size*W/local_group_size, C)
        """
        n, t, h, w, c = size
        return rearrange(
            x,
            "n t (qh ph) (qw pw) c -> (t ph pw) (n qh qw) c",
            n=n,
            t=t,
            qh=h // self.lgs[0],
            ph=self.lgs[0],
            qw=w // self.lgs[0],
            pw=self.lgs[0],
            c=c,
        )

    def rev_permute(self, x, size):
        n, t, h, w, c = size
        return rearrange(
            x,
            "(t ph pw) (n qh qw) c -> n t (qh ph) (qw pw) c",
            n=n,
            t=t,
            qh=h // self.lgs[0],
            ph=self.lgs[0],
            qw=w // self.lgs[0],
            pw=self.lgs[0],
            c=c,
        )


class LocalPermuteModule(object):
    """ Permute the feature map to gather pixels in local groups, and the reverse permutation
    https://github.com/HRNet/HRFormer/blob/main/cls/models/modules/multihead_isa_attention.py
    """
    def __init__(self, local_group_size=7):
        self.lgs = local_group_size
        if not isinstance(self.lgs, (tuple, list)):
            self.lgs = to_2tuple(self.lgs)
        assert len(self.lgs) == 2

    def permute(self, x, size):
        """
        x: (N, H, W, C)
        return: (local_group_size*local_group_size, N*H/local_group_size*W/local_group_size, C)
        """
        n, h, w, c = size
        return rearrange(
            x,
            "n (qh ph) (qw pw) c -> (ph pw) (n qh qw) c",
            n=n,
            qh=h // self.lgs[0],
            ph=self.lgs[0],
            qw=w // self.lgs[0],
            pw=self.lgs[0],
            c=c,
        )

    def rev_permute(self, x, size):
        n, h, w, c = size
        return rearrange(
            x,
            "(ph pw) (n qh qw) c -> n (qh ph) (qw pw) c",
            n=n,
            qh=h // self.lgs[0],
            ph=self.lgs[0],
            qw=w // self.lgs[0],
            pw=self.lgs[0],
            c=c,
        )

class PadBlock(object):
    """ "Make the size of feature map divisible by local group size."""
    """
    https://github.com/HRNet/HRFormer/blob/main/cls/models/modules/multihead_isa_attention.py
    """
    def __init__(self, local_group_size=7):
        self.lgs = local_group_size
        if not isinstance(self.lgs, (tuple, list)):
            self.lgs = to_2tuple(self.lgs)
        assert len(self.lgs) == 2

    def pad_if_needed(self, x, size):
        """
        x: (N, H, W, C)
        """
        n, h, w, c = size
        pad_h = math.ceil(h / self.lgs[0]) * self.lgs[0] - h
        pad_w = math.ceil(w / self.lgs[1]) * self.lgs[1] - w
        if pad_h > 0 or pad_w > 0:  # center-pad the feature on H and W axes
            return F.pad(
                x,
                (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            ) #padding size starting from the last dimension and moving forward.
        return x

    def depad_if_needed(self, x, size):
        """
        x: (N, H, W, C)
        """
        n, h, w, c = size
        pad_h = math.ceil(h / self.lgs[0]) * self.lgs[0] - h
        pad_w = math.ceil(w / self.lgs[1]) * self.lgs[1] - w
        if pad_h > 0 or pad_w > 0:  # remove the center-padding on feature
            return x[:, pad_h // 2 : pad_h // 2 + h, pad_w // 2 : pad_w // 2 + w, :]
        return x

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """https://github.com/rwightman/pytorch-image-models/blob/07379c6d5dbb809b3f255966295a4b03f23af843/timm/models/layers/drop.py#L155
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """https://github.com/rwightman/pytorch-image-models/blob/07379c6d5dbb809b3f255966295a4b03f23af843/timm/models/layers/drop.py#L155
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "drop_prob={}".format(self.drop_prob)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])