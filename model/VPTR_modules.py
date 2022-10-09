import torch
import torch.nn as nn
import torch.nn.functional as F
from .ResNetAutoEncoder import ResnetEncoder, ResnetDecoder
from .VidHRFormer import VidHRFormerNAR, VidHRFormerFAR
from utils.misc import NestedTensor
from utils.position_encoding import PositionEmbeddding1D, PositionEmbeddding2D, PositionEmbeddding3D
import functools

class VPTREnc(nn.Module):
    def __init__(self, img_channels, feat_dim = 528, n_downsampling = 3, padding_type = 'reflect'):
        super().__init__()
        self.feat_dim = feat_dim
        self.encoder = ResnetEncoder(input_nc = img_channels, out_dim = feat_dim, n_downsampling = n_downsampling, padding_type = padding_type)
        
    def forward(self, x):
        """
        Args:
            x --- (N, T, img_channels, H, W)
        Returns:
            feat --- (N, T, 256, 16, 16)
        """
        N, T, _, _, _ = x.shape
        feat = self.encoder(x.flatten(0, 1))
        #feat = self.out_proj(feat)
        _, C, H, W = feat.shape
        feat = feat.reshape(N, T, C, H, W)

        return feat

class VPTRDec(nn.Module):
    def __init__(self, img_channels, feat_dim = 528, n_downsampling = 3, out_layer = 'Tanh', padding_type = 'reflect'):
        super().__init__()
        self.decoder = ResnetDecoder(output_nc = img_channels, feat_dim = feat_dim, n_downsampling = n_downsampling, out_layer = out_layer, padding_type = padding_type)

    def forward(self, feat):
        """
        Args:
            feat --- (N, T, 256, 16, 16)
        """
        N, T, _, _, _ = feat.shape


        out = self.decoder(feat.flatten(0, 1))
        _, C, H, W = out.shape

        return out.reshape(N, T, C, H, W)

class VPTRDisc(nn.Module):
    """
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    Defines a PatchGAN discriminator
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class VPTRFormerNAR(nn.Module):
    def __init__(self, num_past_frames, num_future_frames, encH=8, encW = 8, d_model=528, 
                 nhead=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1, 
                 window_size=4, Spatial_FFN_hidden_ratio=4, TSLMA_flag = False, rpe=True):
        super().__init__()
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.nhead = nhead
        self.d_model = d_model
        self.num_encoder_layers, self.num_decoder_layers = num_encoder_layers, num_decoder_layers

        self.dropout = dropout
        self.window_size = window_size
        self.Spatial_FFN_hidden_ratio = Spatial_FFN_hidden_ratio 

        self.transformer = VidHRFormerNAR((d_model, encH, encW), num_encoder_layers, num_decoder_layers, num_past_frames, num_future_frames,
                    d_model, nhead, window_size = window_size, dropout = dropout, drop_path = dropout, 
                    Spatial_FFN_hidden_ratio = Spatial_FFN_hidden_ratio, dim_feedforward = self.d_model*Spatial_FFN_hidden_ratio, TSLMA_flag = TSLMA_flag, rpe = rpe)
        
        #Init all the pos_embed
        T = num_past_frames+num_future_frames
        pos1d = PositionEmbeddding1D()
        temporal_pos = pos1d(L = T, N = 1, E = self.d_model)[:, 0, :]
        self.register_buffer('temporal_pos', temporal_pos)
        
        pos2d = PositionEmbeddding2D()
        lw_pos = pos2d(N = 1, E = self.d_model, H = window_size, W = window_size)[0, ...].permute(1, 2, 0)
        self.register_buffer('lw_pos', lw_pos)

        pos3d = PositionEmbeddding3D(E = self.d_model, T = T)
        Tlw_pos = pos3d(NestedTensor(torch.empty(T*1, self.d_model, window_size, window_size), None))[0, ...].permute(1, 2, 3, 0)
        self.register_buffer('Tlw_pos', Tlw_pos)

        #init queries of Transformer4
        self.frame_queries = nn.Parameter(torch.randn(num_future_frames, encH, encW, self.d_model), requires_grad = True)

        #projector for the NCEloss
        self.NCE_projector = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                           nn.ReLU(inplace = True),
                                           nn.Linear(self.d_model, self.d_model))
        self._reset_parameters()
        
    def forward(self, past_gt_feat):
        """
        Args:
            past_gt_feats:  (N, T, C, H, W)
        """
        pred, memory = self.transformer(past_gt_feat, self.lw_pos, self.temporal_pos, self.Tlw_pos, self.frame_queries, init_tgt = None)

        return pred
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class VPTRFormerFAR(nn.Module):
    def __init__(self, num_past_frames, num_future_frames, encH=8, encW = 8, d_model=528, 
                 nhead=8, num_encoder_layers=6, dropout=0.1, 
                 window_size=4, Spatial_FFN_hidden_ratio=4,rpe=True):
        super().__init__()
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.nhead = nhead
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers

        self.dropout = dropout
        self.window_size = window_size
        self.Spatial_FFN_hidden_ratio = Spatial_FFN_hidden_ratio 

        self.transformer = VidHRFormerFAR((d_model, encH, encW), num_encoder_layers, num_past_frames, num_future_frames,
                    d_model, nhead, window_size = window_size, dropout = dropout, drop_path = dropout, 
                    Spatial_FFN_hidden_ratio = Spatial_FFN_hidden_ratio, dim_feedforward = self.d_model*Spatial_FFN_hidden_ratio, rpe=rpe)
        
        #Init all the pos_embed
        T = num_past_frames+num_future_frames
        pos1d = PositionEmbeddding1D()
        temporal_pos = pos1d(L = T, N = 1, E = self.d_model)[:, 0, :]
        self.register_buffer('temporal_pos', temporal_pos)
        
        pos2d = PositionEmbeddding2D()
        lw_pos = pos2d(N = 1, E = self.d_model, H = window_size, W = window_size)[0, ...].permute(1, 2, 0)
        self.register_buffer('lw_pos', lw_pos)

        self._reset_parameters()
        
    def forward(self, input_feats):
        """
        Args:
            input_feats:  (N, T, 256, 16, 16)
        """
        pred = self.transformer(input_feats, self.lw_pos, self.temporal_pos)
        
        return pred
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

