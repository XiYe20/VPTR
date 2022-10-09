import torch
import torch.nn as nn
import torch.nn.functional as F

from .VidHRFormer_modules import VidHRFormerEncoder, VidHRFormerBlockEnc
from .VidHRFormer_modules import VidHRformerDecoderNAR, VidHRFormerBlockDecNAR


class VidHRFormerNAR(nn.Module):
    def __init__(self, in_feat_shape, num_encoder_layer, num_decoder_layer, num_past_frames, num_future_frames,
                    embed_dim, num_heads, window_size = 7, dropout = 0., drop_path = 0., Spatial_FFN_hidden_ratio = 4, 
                    dim_feedforward = 512, TSLMA_flag = False, rpe = True):
        super().__init__()
        self.in_C, self.H, self.W = in_feat_shape
        self.embed_dim = embed_dim
        #self.conv_proj = nn.Identity() if self.in_C == self.embed_dim else self.feat_proj()
        #self.conv_proj_rev = nn.Identity() if self.in_C == self.embed_dim else self.feat_proj_rev()

        self.num_encoder_layer = num_encoder_layer
        self.num_decoder_layer = num_decoder_layer
        self.num_heads = num_heads

        self.encoder = VidHRFormerEncoder(VidHRFormerBlockEnc(self.H, self.W, embed_dim, num_heads, window_size, dropout, drop_path, Spatial_FFN_hidden_ratio, dim_feedforward, rpe=rpe), 
                                        num_encoder_layer, nn.LayerNorm(embed_dim))
        self.decoder = VidHRformerDecoderNAR(VidHRFormerBlockDecNAR(self.H, self.W, embed_dim, num_heads, window_size, dropout, drop_path, Spatial_FFN_hidden_ratio, dim_feedforward, TSLMA_flag, rpe=rpe),
                                        num_decoder_layer, nn.LayerNorm(embed_dim), return_intermediate=False)
                                
    def forward(self, src, local_window_pos_embed, temporal_pos_embed, TS_local_pos_embed, query_pos, init_tgt = None):
        """
        Args:
            src: feature extracted by the backbone, (N, Tp, in_C, H, W)
            local_window_pos_embed: (window_size, window_size, embed_dim)
            temporal_pos_embed: (Tp+Tf, embed_dim)
            TS_local_pos_embed: (Tp+Tf, window_size, window_size, embed_dim)
            query_pos: (Tf, H, W, embed_dim)
        Return:
            out: (N, Tf, H, W, embed_dim), for the next layer query_pos init
            out_proj: (N, Tf, in_C, H, W), final output feature for the decoder
            memory: (N, Tp, H, W, embed_dim)
        """
        N, Tp, _, _, _ = src.shape
        #src = self.conv_proj(src.view(N*Tp, self.in_C, self.H, self.W)).view(N, Tp, self.embed_dim, self.H, self.W)
        src = src.permute(0, 1, 3, 4, 2)
        memory = self.encoder(src, local_window_pos_embed, temporal_pos_embed[0:Tp, ...])
        #pred_query = self.FBP(memory, src) #(N, Tf, H, W, embed_dim)
        query_pos = query_pos.unsqueeze(0).repeat(N, 1, 1, 1, 1)# + pred_query

        init_tgt = torch.zeros_like(query_pos, requires_grad = False) #init as zeros
        out = self.decoder(init_tgt, query_pos, memory, local_window_pos_embed, temporal_pos_embed[Tp:, ...], TS_local_pos_embed, temporal_pos_embed[0:Tp, ...]) #(N, Tf, H, W, embed_dim)

        out = F.relu_(out.permute(0, 1, 4, 2, 3))

        return out, memory


class VidHRFormerFAR(nn.Module):
    def __init__(self, in_feat_shape, num_encoder_layer, num_past_frames, num_future_frames,
                    embed_dim, num_heads, window_size = 7, dropout = 0., drop_path = 0., Spatial_FFN_hidden_ratio = 4, dim_feedforward = 512, rpe = True):
        super().__init__()
        self.in_C, self.H, self.W = in_feat_shape
        self.embed_dim = embed_dim
        #self.conv_proj = nn.Identity() if self.in_C == self.embed_dim else self.feat_proj()
        #self.conv_proj_rev = nn.Identity() if self.in_C == self.embed_dim else self.feat_proj_rev()

        self.num_encoder_layer = num_encoder_layer
        self.num_heads = num_heads

        self.encoder = VidHRFormerEncoder(VidHRFormerBlockEnc(self.H, self.W, embed_dim, num_heads, window_size, dropout, drop_path, Spatial_FFN_hidden_ratio, dim_feedforward, far = True, rpe=rpe), 
                                        num_encoder_layer, nn.LayerNorm(embed_dim))
        
    def forward(self, input_feat, local_window_pos_embed, temporal_pos_embed):
        """
        Args:
            past_gt_feats:  (N, T, 528, 8, 8)
            future_frames: (N, T, 528, 8, 8) or None
            local_window_pos_embed: (window_size, window_size, embed_dim)
            temporal_pos_embed: (Tp+Tf, embed_dim)
        Return:
            out: (N, Tf, H, W, embed_dim), for the next layer query_pos init
            out_proj: (N, Tf, in_C, H, W), final output feature for the decoder
            memory: (N, Tp, H, W, embed_dim)
        """
        src = input_feat
        N, T, _, _, _ = src.shape
        src = src.permute(0, 1, 3, 4, 2)
        pred = self.encoder(src, local_window_pos_embed, temporal_pos_embed[0:T, ...])
        pred = F.relu_(pred.permute(0, 1, 4, 2, 3))
        
        return pred