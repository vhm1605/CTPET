import torch
import torch.nn as nn
import re
from .helpers import PerceiverResampler

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
    
    
# class Resampler(nn.Module):
#     def __init__(self, embedding_dim=4096, vis_dim=512, perceiver_num=64):
#         super().__init__()
#         self.perceiver = PerceiverResampler(dim=vis_dim, num_latents=perceiver_num)
#         self.fc = nn.Linear(vis_dim, embedding_dim)
        
#     def forward(self, x, return_attn=False):
#         B, D, H, W, C = x.shape
#         x = x.view(B, 1, 1, D * H * W, C)
        
#         if return_attn:
#             x, attn = self.perceiver(x, return_attn)
#         else:
#             x = self.perceiver(x, return_attn)
            
#         x = x.view(B, -1, x.shape[3])
#         x = self.fc(x)
        
#         if return_attn:
#             return x, attn
#         return x 
    
class Resampler(nn.Module):
    def __init__(self, embedding_dim=4096, vis_dim=512):
        super().__init__()
        self.perceiver_g = PerceiverResampler(dim=vis_dim, num_latents=128)
        self.fc_g = nn.Linear(vis_dim, embedding_dim)
        
        # self.perceiver_fpet = PerceiverResampler(dim=vis_dim, num_latents=64)
        # self.fc_fpet = nn.Linear(vis_dim, embedding_dim)
        
        # self.perceiver_fct = PerceiverResampler(dim=vis_dim, num_latents=64)
        # self.fc_fct = nn.Linear(vis_dim, embedding_dim)

        self.fc_fpet = nn.Sequential(
            nn.Linear(vis_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # focal_ct: giữ nguyên số token P
        self.fc_fct = nn.Sequential(
            nn.Linear(vis_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, x, return_attn=False, mode="global"):
        if mode == "global":
            B, D, H, W, C = x.shape
            x = x.view(B, 1, 1, D * H * W, C)
            
            if return_attn:
                x, attn = self.perceiver_g(x, return_attn)
            else:
                x = self.perceiver_g(x, return_attn)
                
            x = x.view(B, -1, x.shape[3])
            x = self.fc_g(x)
            
            if return_attn:
                return x, attn
            return x 
        elif mode == "focal_pet":
            # x: (B, P, N=vis_dim)
            x = self.fc_fpet(x)             # (B, P, 4096)
            return x

        elif mode == "focal_ct":
            # x: (B, P, N=vis_dim)
            x = self.fc_fct(x)              # (B, P, 4096)
            return x
        else:
            return None
    
# class Resampler(nn.Module):
#     def __init__(self, embedding_dim=4096, vis_dim=512, stride=(2, 3, 3)):
#         super().__init__()
        
#         # Conv3D giảm kích thước D,H,W
#         self.conv3d = nn.Conv3d(
#             in_channels=vis_dim, 
#             out_channels=vis_dim*4, 
#             kernel_size=(3, 3, 3), 
#             stride=stride, 
#             padding=1
#         )
        
#         # Linear projection sang embedding_dim cho LLM
#         self.fc = nn.Linear(vis_dim*4, embedding_dim)
        
#     def forward(self, x):
#         """
#         Args:
#             x: (B, D, H, W, C) - Visual embedding từ CLIP
#         Returns:
#             out: (B, N, embedding_dim) - N là số tokens sau Conv3D flatten
#         """
#         B, D, H, W, C = x.shape
        
#         # Đổi thành (B, C, D, H, W) cho Conv3D
#         x = x.permute(0, 4, 1, 2, 3)
        
#         # Conv3D downsample
#         x = self.conv3d(x)  # (B, vis_dim*4, D', H', W')
#         B, C_conv, D_new, H_new, W_new = x.shape
        
#         # Flatten tokens: (B, N, C_conv)
#         x = x.view(B, C_conv, -1).transpose(1, 2)
        
#         # Linear projection: (B, N, embedding_dim)
#         x = self.fc(x)
        
#         return x

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    print(projector_type, config.mm_hidden_size, config.hidden_size)
    if projector_type == 'linear':
        # return Resampler(embedding_dim=config.hidden_size, vis_dim=config.mm_hidden_size)
        return Resampler(embedding_dim=config.hidden_size, vis_dim=config.mm_hidden_size)
        # return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
