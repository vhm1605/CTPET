import wandb
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from ct_clip import CTCLIP, TextTransformer
from CTCLIPTrainer import CTClipTrainer
from ctvit import CTViT
import os

wandb.login(key="c0bf463d253eb9147fbe555216398f2838fe517c")

# Khởi tạo wandb
wandb.init(
    project="PET_CLIP",
    name="exp_ct_clip_v1",   
    entity="dacthai2807"
)

# Load tokenizer và text encoder
pretrain = 'vinai/phobert-base'
tokenizer = AutoTokenizer.from_pretrained(pretrain, do_lower_case=True)
text_encoder = AutoModel.from_pretrained(pretrain)

# Khởi tạo image encoder
image_encoder = CTViT(
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

# Khởi tạo CLIP model
clip = CTCLIP(
    image_encoder=image_encoder,
    text_encoder=text_encoder,
    dim_text=768,
    dim_image=294912,
    dim_latent=512,
    extra_latent_projection=False,
    use_mlm=False,
    downsample_image_embeds=False,
    use_all_token_embeds=False
)

# Load pretrained checkpoint nếu có
import torch
from collections import OrderedDict

checkpoint_path = '/home/jovyan/shared/tienhuu060102/data-petct/pretrained_weights/CT-CLIP_v2_PhoBert.pt'
state_dict = torch.load(checkpoint_path)

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("module.", "")
    new_state_dict[new_key] = v

clip.load_state_dict(new_state_dict)

# Dùng wandb logger thay vì custom Logger
trainer = CTClipTrainer(
    clip,
    root='/home/jovyan/shared/tienhuu060102/data-petct/PET_report_paired_fixed',
    comparation_path=None,
    batch_size=8,
    tokenizer=tokenizer,
    results_folder="results/CT_CLIP",
    num_train_steps=100001,
    num_workers=2, 
)

trainer.train()

wandb.finish()
