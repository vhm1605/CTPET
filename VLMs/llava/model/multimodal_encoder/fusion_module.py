
# # import torch
# # import torch.nn as nn
# from .ctvit import CTViT

# # class PETCTFusion(nn.Module):
# #     def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
# #         super().__init__()
        
# #         self.pet_enc = CTViT(
# #             dim=512,
# #             codebook_size=8192,
# #             image_size=480,
# #             patch_size=20,
# #             temporal_patch_size=10,
# #             spatial_depth=4,
# #             temporal_depth=4,
# #             dim_head=32,
# #             heads=8
# #         )

# #         pet_ckpt = '/data/backup_sde1/PET-CT-report/ckpt/pet_emb/ctvit.76000.pt'
# #         checkpoint = torch.load(pet_ckpt, map_location='cpu')
# #         from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
# #         consume_prefix_in_state_dict_if_present(checkpoint, prefix="module.")
# #         self.pet_enc.load_state_dict(checkpoint) 

# #         for param in self.pet_enc.parameters():
# #             param.requires_grad = False

# #         self.pet_enc.eval() 

# #         self.ct_enc = CTViT(
# #             dim=512,
# #             codebook_size=8192,
# #             image_size=480,
# #             patch_size=20,
# #             temporal_patch_size=10,
# #             spatial_depth=4,
# #             temporal_depth=4,
# #             dim_head=32,
# #             heads=8
# #         )

# #         ct_ckpt = '/data/backup_sde1/PET-CT-report/ckpt/ct_emb/ctvit.76000.pt'
# #         checkpoint = torch.load(ct_ckpt, map_location='cpu')
# #         from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
# #         consume_prefix_in_state_dict_if_present(checkpoint, prefix="module.")
# #         self.ct_enc.load_state_dict(checkpoint) 

# #         for param in self.ct_enc.parameters():
# #             param.requires_grad = False

# #         self.ct_enc.eval() 

# #         self.embed_dim = embed_dim
# #         self.num_heads = num_heads

# #         # Cross-Attention module
# #         self.cross_attn = nn.MultiheadAttention(
# #             embed_dim=embed_dim,
# #             num_heads=num_heads,
# #             dropout=dropout,
# #             batch_first=True  # Important for (B, N, C)
# #         )

# #         # Optional: LayerNorm before and after attention
# #         self.norm1 = nn.LayerNorm(embed_dim)
# #         self.norm2 = nn.LayerNorm(embed_dim)

# #         # Feed Forward after cross-attn
# #         self.ffn = nn.Sequential(
# #             nn.Linear(embed_dim, embed_dim * 4),
# #             nn.GELU(),
# #             nn.Linear(embed_dim * 4, embed_dim),
# #             nn.Dropout(dropout)
# #         )

# #     def forward(self, pet_img, ct_img):
# #         with torch.no_grad(): 
# #             pet_emb = self.pet_enc(pet_img)
# #             ct_emb = self.ct_enc(ct_img)
        
# #         B, D, H, W, C = pet_emb.shape

# #         # Reshape (B, D, H, W, C) --> (B, N, C)
# #         pet_tokens = pet_emb.contiguous().view(B, D * H * W, C)
# #         ct_tokens  = ct_emb.contiguous().view(B, D * H * W, C)

# #         # Normalize inputs before attention
# #         pet_norm = self.norm1(pet_tokens)
# #         ct_norm  = self.norm1(ct_tokens)

# #         # Cross Attention: Query = PET, Key/Value = CT
# #         attn_output, _ = self.cross_attn(query=pet_norm, key=ct_norm, value=ct_norm)

# #         # Skip connection from PET tokens
# #         fused = pet_tokens + attn_output

# #         # Feed Forward + second skip connection
# #         fused_norm = self.norm2(fused)
# #         fused = fused + self.ffn(fused_norm)
        
# #         fused = fused.contiguous().view(B, D, H, W, C)
        
# #         return fused 





# import wandb
# # from ctvit import CTViT
# import torch
# import torch.nn as nn
# from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

# import sys
# # sys.path.append("../")
# from .cosmos1.models.tokenizer.networks import TokenizerConfigs, TokenizerModels
# import torch
# from tqdm import tqdm
# import os
# import yaml
# import wandb
# import random
# import numpy as np
# import os 
# # os.environ["CUDA_VISIBLE_DEVICES"] = "3"




# model_class = TokenizerModels.CV.value
# config = TokenizerConfigs.CV8x8x8.value






# class PETCTFusion(nn.Module):
#     def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
#         super().__init__()
        
#         # self.pet_enc = CTViT(
#         #     dim=512,
#         #     codebook_size=8192,
#         #     image_size=480,
#         #     patch_size=20,
#         #     temporal_patch_size=10,
#         #     spatial_depth=4,
#         #     temporal_depth=4,
#         #     dim_head=32,
#         #     heads=8
#         # )

#         ###########
#         pet_cosmos_path = '/data/backup_sde1/cosmos/cosmos_10.pth'
#         model_class = TokenizerModels.CV.value
#         config = TokenizerConfigs.CV8x8x8.value
#         self.pet_enc = model_class(**config)
#         # model_weight = torch.jit.load(pet_cosmos_path)
#         # state_dict = model_weight.state_dict()
#         # del model_weight
#         # torch.cuda.empty_cache()
#         # self.pet_enc.load_state_dict(state_dict, strict=True)
#         state = torch.load(pet_cosmos_path, map_location="cpu")
#         self.pet_enc.load_state_dict(state)
#         self.pet_enc = self.pet_enc.encoder
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.pet_enc = self.pet_enc.to(device)
        
#         #####
#         # pet_ckpt = '/home/jovyan/shared/tienhuu060102/data-petct/pretrained_weights/pet_emb/ctvit.76000.pt'
#         # checkpoint = torch.load(pet_ckpt, map_location='cpu')
#         # from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
#         # consume_prefix_in_state_dict_if_present(checkpoint, prefix="module.")
#         # self.pet_enc.load_state_dict(checkpoint) 

#         for param in self.pet_enc.parameters():
#             param.requires_grad = False

#         self.pet_enc.eval() 

#         # self.ct_enc = CTViT(
#         #     dim=512,
#         #     codebook_size=8192,
#         #     image_size=480,
#         #     patch_size=20,
#         #     temporal_patch_size=10,
#         #     spatial_depth=4,
#         #     temporal_depth=4,
#         #     dim_head=32,
#         #     heads=8
#         # )


#         ###########
#         ct_cosmos_path = '/data/backup_sde1/cosmos/cosmos_ct_model_epoch_8.pth'
#         model_class = TokenizerModels.CV.value
#         config = TokenizerConfigs.CV8x8x8.value
#         self.ct_enc = model_class(**config)
#         # model_weight = torch.jit.load(ct_cosmos_path)
#         # state_dict = model_weight.state_dict()
#         # del model_weight
#         # torch.cuda.empty_cache()
#         # self.ct_enc.load_state_dict(state_dict, strict=True)
#         state = torch.load(ct_cosmos_path, map_location="cpu")
#         self.ct_enc.load_state_dict(state)
#         self.ct_enc = self.ct_enc.encoder
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.ct_enc = self.ct_enc.to(device)
#         #####
#         # ct_ckpt = '/home/jovyan/shared/tienhuu060102/data-petct/pretrained_weights/ct_emb/ctvit.76000.pt'
#         # checkpoint = torch.load(ct_ckpt, map_location='cpu')
#         # from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
#         # consume_prefix_in_state_dict_if_present(checkpoint, prefix="module.")
#         # self.ct_enc.load_state_dict(checkpoint) 

#         for param in self.ct_enc.parameters():
#             param.requires_grad = False

#         self.ct_enc.eval() 

#         self.embed_dim = embed_dim
#         self.num_heads = num_heads

#         # Cross-Attention module
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True  # Important for (B, N, C)
#         )

#         # Optional: LayerNorm before and after attention
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.norm3 = nn.LayerNorm(embed_dim)

#         # Feed Forward after cross-attn
#         self.ffn = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * 4),
#             nn.GELU(),
#             nn.Linear(embed_dim * 4, embed_dim),
#             nn.Dropout(dropout)
#         )
        
#         self.fuse_mlp = nn.Linear(embed_dim * 2, embed_dim)

#         self.pet_conv3d = nn.Conv3d(
#                 in_channels=16,
#                 out_channels=embed_dim,
#                 kernel_size=(4, 2, 2),
#                 stride=(4, 2, 2),
#                 padding=0
#             )
#         self.ct_conv3d = nn.Conv3d(
#                 in_channels=16,
#                 out_channels=embed_dim,
#                 kernel_size=(4, 2, 2),
#                 stride=(4, 2, 2),
#                 padding=0
#             )
        
#         print("-------set up model done------")
#     def forward(self, pet_img, ct_img):
#         with torch.no_grad(): 
#             pet_emb = self.pet_enc(pet_img)
#             ct_emb = self.ct_enc(ct_img)

#         B, C, D, H, W = pet_emb.shape
#         pet_emb = self.pet_conv3d(pet_emb)
#         ct_emb = self.ct_conv3d(ct_emb)

#         _, c, d, h, w = pet_emb.shape
#         pet_emb = pet_emb.permute(0, 2, 3, 4, 1)
#         ct_emb = ct_emb.permute(0, 2, 3, 4, 1)
#         pet_tokens = pet_emb.contiguous().view(B, d * h * w, c)
#         ct_tokens  = ct_emb.contiguous().view(B, d * h * w, c)

#         # Normalize inputs before attention
#         pet_norm = self.norm1(pet_tokens)
#         ct_norm  = self.norm2(ct_tokens)

#         # Cross Attention: Query = PET, Key/Value = CT
#         attn_output, _ = self.cross_attn(query=pet_norm, key=ct_norm, value=ct_norm)

#         # Skip connection from PET tokens
#         fused = pet_tokens + attn_output

#         # Feed Forward + second skip connection
#         fused_norm = self.norm3(fused)
#         fused = fused + self.ffn(fused_norm)
        
#         fused = fused.contiguous().view(B, d, h, w, c)
#         print(fused.shape)
#         return fused  # (B, N, C)



# if __name__ == "__main__":
#     ckpt = "/data/backup_sde1/cosmos/CTClip.80000.pt"

#     model = PETCTFusion()
#     sd = torch.load(ckpt, map_location="cpu")

#     if "state_dict" in sd:
#         sd = sd["state_dict"]

#     result = model.load_state_dict(sd, strict=False)

#     total = len(model.state_dict())
#     success = total - len(result.missing_keys)

#     print("success:", success)
#     print("missing:", len(result.missing_keys))
#     print("unexpected:", len(result.unexpected_keys))
#     print(f"loaded: {success}/{total}")



import torch
import torch.nn as nn
from .ctvit import CTViT

class PETCTFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.pet_enc = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8
        )

        pet_ckpt = '/data/backup_sde1/PET-CT-report/ckpt/pet_emb/ctvit.76000.pt'
        checkpoint = torch.load(pet_ckpt, map_location='cpu')
        from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
        consume_prefix_in_state_dict_if_present(checkpoint, prefix="module.")
        self.pet_enc.load_state_dict(checkpoint) 

        for param in self.pet_enc.parameters():
            param.requires_grad = False

        self.pet_enc.eval() 

        self.ct_enc = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8
        )

        ct_ckpt = '/data/backup_sde1/PET-CT-report/ckpt/ct_emb/ctvit.76000.pt'
        checkpoint = torch.load(ct_ckpt, map_location='cpu')
        from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
        consume_prefix_in_state_dict_if_present(checkpoint, prefix="module.")
        self.ct_enc.load_state_dict(checkpoint) 

        for param in self.ct_enc.parameters():
            param.requires_grad = False

        self.ct_enc.eval() 

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Cross-Attention module
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Important for (B, N, C)
        )

        # Optional: LayerNorm before and after attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed Forward after cross-attn
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, pet_img, ct_img):
        with torch.no_grad(): 
            pet_emb = self.pet_enc(pet_img)
            ct_emb = self.ct_enc(ct_img)
        
        B, D, H, W, C = pet_emb.shape

        # Reshape (B, D, H, W, C) --> (B, N, C)
        pet_tokens = pet_emb.contiguous().view(B, D * H * W, C)
        ct_tokens  = ct_emb.contiguous().view(B, D * H * W, C)

        # Normalize inputs before attention
        pet_norm = self.norm1(pet_tokens)
        ct_norm  = self.norm1(ct_tokens)

        # Cross Attention: Query = PET, Key/Value = CT
        attn_output, _ = self.cross_attn(query=pet_norm, key=ct_norm, value=ct_norm)

        # Skip connection from PET tokens
        fused = pet_tokens + attn_output

        # Feed Forward + second skip connection
        fused_norm = self.norm2(fused)
        fused = fused + self.ffn(fused_norm)
        
        fused = fused.contiguous().view(B, D, H, W, C)
        
        return fused 