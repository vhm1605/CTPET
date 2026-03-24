#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
import numpy as np
import json
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape

import re
import unicodedata
class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            print('initialize_vision_modules: build_vision_tower: 35')
            print("-" * 100 ,'\nconfig: ', config, '\n', "-" * 100 )
            self.vision_tower = build_vision_tower(config, delay_load=True)
            # config.mm_hidden_size = self.vision_tower.hidden_size

            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        print('initialize_vision_modules')
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            print('get_vision_tower is None 61')
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            # vision_tower.load_model()
        
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        print (f"Using mm_hidden_size: {self.config.mm_hidden_size}")
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        # if getattr(self, 'mm_projector', None) is None:
        print('initialize_vision_modules: build_vision_projector: 81')
        self.mm_projector = build_vision_projector(self.config)

        if 'unpad' in mm_patch_merge_type:
            embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
            self.image_newline = nn.Parameter(
                torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
            )
        # else:
            # In case it is frozen by LoRA
        for p in self.mm_projector.parameters():
            p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            print('LOADING PRETRAINED MM MLP ADAPTER FROM: ', pretrain_mm_mlp_adapter)
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_projector(self):
        return self.get_model().mm_projector

    # def encode_images(self, images):
    #     if 'PET' in images and 'CT' in images:
    #         pet_images, ct_images = images['PET'], images['CT']
    #         image_features = self.get_model().get_vision_tower()(pet_images, ct_images)
    #     else:
    #         images = images['data']
    #         image_features = self.get_model().get_vision_tower()(images)
    #     # image_features = image_features.view(image_features.shape[0], image_features.shape[1], -1)  # Reshape the output tensor
    #     image_features = self.get_model().mm_projector(image_features)

    #     return image_features

    # def encode_images(self, images):
    #     vision_tower = self.get_model().get_vision_tower()

    #     if 'PET' in images and 'CT' in images:
    #         pet_images = images['PET']   # (B, c, t, H, W)
    #         ct_images  = images['CT']
    #         ct_segs = images['CT_SEG']

    #         device = ct_images.device

    #         # 1. PET-guided voxel mask

    #         voxel_mask = vision_tower.pet_enc.create_pet_mask(pet_images)
            



    #         with open('/home/thaind/labels.json', 'r') as f:
    #             labels_map = json.load(f)


    #         # 2. Extract PET patch embeddings (list of (Ni, 512))
    #         pet_patch_list = vision_tower.pet_enc.extract_patch_embeddings(
    #             pet_images, voxel_mask, L=2
    #         )

    #         # 3. Extract CT patch embeddings (list of (Ni, 512))
    #         ct_patch_list, patch_labels_list, patch_indices_list  = vision_tower.ct_enc.extract_patch_embeddings_CT(
    #             ct_images, voxel_mask, ct_segs, L=2
    #         )

    #         # # 3. Pad + resample
    #         # P = self.get_model().mm_projector.perceiver_fpet.num_latents
    #         # D = self.get_model().mm_projector.fc_fpet.out_features
            
    #         # pet_batch_tokens, ct_batch_tokens = [], []

    #         # for pet_embs, ct_embs in zip(pet_patch_list, ct_patch_list):
    #         #     pet_batch_tokens.append(
    #         #         torch.zeros(P, D, device=device)
    #         #         if pet_embs.numel() == 0
    #         #         else self.get_model().mm_projector(
    #         #             pet_embs.unsqueeze(0), mode="focal_pet"
    #         #         ).squeeze(0)
    #         #     )

    #         #     ct_batch_tokens.append(
    #         #         torch.zeros(P, D, device=device)
    #         #         if ct_embs.numel() == 0
    #         #         else self.get_model().mm_projector(
    #         #             ct_embs.unsqueeze(0), mode="focal_ct"
    #         #         ).squeeze(0)
    #         #     )

    #         # pet_focal_feat = torch.stack(pet_batch_tokens, dim=0)  # (B, P, D)
    #         # ct_focal_feat = torch.stack(ct_batch_tokens, dim=0)  # (B, P, D)
    #         global_feat = vision_tower(pet_images, ct_images)
    #         global_feat = self.get_model().mm_projector(global_feat, mode="global")
    #         # pet_focal_feat = torch.stack(pet_batch_tokens, dim=0)  # (B, P, D)
    #         # ct_focal_feat = torch.stack(ct_batch_tokens, dim=0)  # (B, P, D)

    #         B = ct_images.shape[0]
    #         region_json_texts = []

    #         for i in range(B):
    #             sample_json = {
    #                 "global": {
    #                     "description": "Global feature extracted from whole CT and PET volumes",
    #                     "CT/PET": global_feat[i]
                         

    #                 },
    #                 "regions": {}
    #             }
    #             num_regions = len(patch_labels_list[i])

    #             for k in range(num_regions):
    #                 region_name = f"region_{k+1}"

    #                 # label ids của patch k
    #                 label_ids = patch_labels_list[i][k]
    #                 if torch.is_tensor(label_ids):
    #                     label_ids = label_ids.detach().cpu().tolist()

    #                 # map sang tên bộ phận
    #                 organ_names = [
    #                     labels_map.get(str(lbl), labels_map.get(lbl, f"{lbl}"))
    #                     for lbl in label_ids
    #                 ]

    #                 # # vị trí patch
    #                 # patch_idx = patch_indices_list[i][k]
    #                 # if torch.is_tensor(patch_idx):
    #                 #     t_idx, h_idx, w_idx = patch_idx.detach().cpu().tolist()
    #                 # else:
    #                 #     t_idx, h_idx, w_idx = patch_idx

    #                 # mô tả cơ bản
                    
    #                 organs_str = ", ".join(organ_names)
    #                 description = (
    #                     f"Đây là vùng SUV cao, có khả năng bất thường, liên quan đến các bộ phận: {organs_str}"
    #                 )

    #                 sample_json["regions"][region_name] = {
    #                     "description": description,
    #                 "pet_patch_emb": pet_patch_list[i][k],
    #                 "ct_patch_emb": ct_patch_list[i][k],
    #                 }

    #             region_json_text = (sample_json)
    #             region_json_texts.append(region_json_text)

    #         return region_json_texts
    #      #   return torch.cat([global_feat, pet_focal_feat, ct_focal_feat], dim=1)

    #     else:
    #         images = images['data']
    #         image_features = vision_tower(images)
    #         image_features = self.get_model().mm_projector(image_features)
    #         return image_features




    def encode_images(self, images):
        vision_tower = self.get_model().get_vision_tower()

        if 'PET' in images and 'CT' in images:
            pet_images = images['PET']
            ct_images  = images['CT']
            ct_segs    = images['CT_SEG']
            paths = images['PATHS']

            device = ct_images.device

            voxel_mask = vision_tower.pet_enc.create_pet_mask(pet_images)

            with open('/home/thaind/anonymous_project_copy/labels.json', 'r') as f:
                part_groups = json.load(f)

            part_groups = {
                group_name: {int(k): v for k, v in group.items()}
                for group_name, group in part_groups.items()
            }

            valid_label_sets = {
                group_name: set(group_dict.keys()) | {0}
                for group_name, group_dict in part_groups.items()
            }

            labels_map = {}
            for group_dict in part_groups.values():
                for label_id, organ_name in group_dict.items():
                    labels_map[label_id] = organ_name

            filtered_ct_segs = []

            for seg, path in zip(ct_segs, paths):
                path_lower = path.lower()

                if "head_neck" in path_lower:
                    region = "head_neck"
                elif "chest" in path_lower:
                    region = "chest"
                elif "abdomen_pelvis" in path_lower:
                    region = "abdomen_pelvis"
                else:
                    raise ValueError(f"Unknown region in path: {path}")

                valid_ids = torch.tensor(
                    sorted(valid_label_sets[region]),
                    device=seg.device,
                    dtype=seg.dtype
                )

                keep_mask = torch.isin(seg, valid_ids)
                seg_filtered = torch.where(keep_mask, seg, torch.zeros_like(seg))
                filtered_ct_segs.append(seg_filtered)

            ct_segs = torch.stack(filtered_ct_segs, dim=0)

            pet_patch_tensor, pet_labels_per_sample, pet_region_patch_indices_per_sample = \
                vision_tower.pet_enc.extract_patch_embeddings_label(
                    pet_images, voxel_mask, ct_segs, L=2
                )

            ct_patch_tensor, ct_labels_per_sample, ct_region_patch_indices_per_sample = \
                vision_tower.ct_enc.extract_patch_embeddings_label(
                    ct_images, voxel_mask, ct_segs, L=2
                )

            print("pet_images shape:", pet_images.shape)
            print("ct_images shape:", ct_images.shape)
            print("len(pet_labels_per_sample):", len(pet_labels_per_sample))
            print("len(ct_labels_per_sample):", len(ct_labels_per_sample))

            # global feature
            global_feat = vision_tower(pet_images, ct_images)
            global_feat = self.get_model().mm_projector(global_feat, mode="global")

            B = ct_images.shape[0]

            pet_region_embeds_per_sample = []
            ct_region_embeds_per_sample = []
            region_texts_per_sample = []

            for i in range(B):
                cur_pet_regions = []
                cur_ct_regions = []
                cur_region_texts = []

                pet_labels_i = pet_labels_per_sample[i]
                ct_labels_i  = ct_labels_per_sample[i]

                pet_embs_i = self.get_model().mm_projector(
                    pet_patch_tensor[i].unsqueeze(1),
                    mode="focal_pet"
                ).squeeze(1)

                ct_embs_i = self.get_model().mm_projector(
                    ct_patch_tensor[i].unsqueeze(1),
                    mode="focal_ct"
                ).squeeze(1)

                # map label_id -> embedding
                pet_label_to_emb = {}
                for k, lbl_id in enumerate(pet_labels_i.tolist()):
                    pet_label_to_emb[lbl_id] = pet_embs_i[k]

                ct_label_to_emb = {}
                for k, lbl_id in enumerate(ct_labels_i.tolist()):
                    ct_label_to_emb[lbl_id] = ct_embs_i[k]

                # chỉ lấy phần giao nhau để PET/CT region luôn khớp nhau
                common_labels = sorted(set(pet_label_to_emb.keys()) & set(ct_label_to_emb.keys()))
                print(len(common_labels), len(set(pet_label_to_emb.keys())), len(set(ct_label_to_emb.keys())))

                max_regions = 100
                common_labels = common_labels[:max_regions]

                for lbl_id in common_labels:
                    organ_name = labels_map.get(lbl_id, f"{lbl_id}")

                    # (4096,) -> (1, 1, 4096) để giữ format cũ
                    pet_emb = pet_label_to_emb[lbl_id].unsqueeze(0).unsqueeze(0)
                    ct_emb  = ct_label_to_emb[lbl_id].unsqueeze(0).unsqueeze(0)

                    cur_pet_regions.append(pet_emb)
                    cur_ct_regions.append(ct_emb)

                    # chỉ lưu tên cơ quan để bên dưới build prompt
                    cur_region_texts.append(organ_name)

                pet_region_embeds_per_sample.append(cur_pet_regions)
                ct_region_embeds_per_sample.append(cur_ct_regions)
                region_texts_per_sample.append(cur_region_texts)

            print("global_feat:", global_feat.shape)

            if len(pet_region_embeds_per_sample) > 0 and len(pet_region_embeds_per_sample[0]) > 0:
                print("pet_region_embeds_per_sample[0][0]:", pet_region_embeds_per_sample[0][0].shape)

            if len(ct_region_embeds_per_sample) > 0 and len(ct_region_embeds_per_sample[0]) > 0:
                print("ct_region_embeds_per_sample[0][0]:", ct_region_embeds_per_sample[0][0].shape)

            if len(region_texts_per_sample) > 0 and len(region_texts_per_sample[0]) > 0:
                print("region_texts_per_sample[0][0]:", region_texts_per_sample[0][0])

            return (
                global_feat,
                pet_region_embeds_per_sample,
                ct_region_embeds_per_sample,
                region_texts_per_sample,
            )

        raise NotImplementedError("Unsupported image format")


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # encode_images trả về:
        # global_feat: (B, G, 4096)
        # pet_region_embeds_per_sample[b][k]: (1, 1, 4096)
        # ct_region_embeds_per_sample[b][k]:  (1, 1, 4096)
        # region_texts_per_sample[b][k]: tên cơ quan
        global_feat, pet_region_embeds_per_sample, ct_region_embeds_per_sample, region_texts_per_sample = self.encode_images(images)

        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(self.get_model(), "tokenizer", None)

        print("tokenizer =", tokenizer)
        print("type(tokenizer) =", type(tokenizer))
        print("class name =", tokenizer.__class__.__name__ if tokenizer is not None else None)
        print("module =", tokenizer.__class__.__module__ if tokenizer is not None else None)

        device = input_ids.device
        embed_tokens = self.get_model().embed_tokens
        embed_dim = embed_tokens.weight.shape[1]
        embed_dtype = embed_tokens.weight.dtype

        # =========================================================
        # Build multimodal block cho từng sample:
        # [text global][global_feat]
        # [text organ1][pet1][ct1]
        # [text organ2][pet2][ct2]
        # ...
        # =========================================================
        image_features = []
        B = len(region_texts_per_sample)

        for b in range(B):
            cur_blocks = []

            # ===== global embedding =====
            cur_blocks.append(global_feat[b].to(device))

            # ===== global text =====
            if tokenizer is not None:
                global_text = "là thông tin của toàn bộ ảnh PET/CT."
                global_text_ids = tokenizer(
                    global_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).input_ids.to(device)

                global_text_embeds = embed_tokens(global_text_ids[0])
                cur_blocks.append(global_text_embeds)

            num_regions = len(region_texts_per_sample[b])

            for k in range(num_regions):
                organ_name = region_texts_per_sample[b][k]

                # ===== PET region embedding =====
                pet_feat = pet_region_embeds_per_sample[b][k].squeeze(0).to(device)
                cur_blocks.append(pet_feat)

                # ===== CT region embedding =====
                ct_feat = ct_region_embeds_per_sample[b][k].squeeze(0).to(device)
                cur_blocks.append(ct_feat)

                # ===== region text =====
                if tokenizer is not None:
                    region_text = f"là thông tin PET và CT của vùng {organ_name}."
                    text_ids = tokenizer(
                        region_text,
                        add_special_tokens=False,
                        return_tensors="pt"
                    ).input_ids.to(device)

                    text_embeds = embed_tokens(text_ids[0])
                    cur_blocks.append(text_embeds)

            cur_mm = torch.cat(cur_blocks, dim=0)
            image_features.append(cur_mm)

        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device
            )

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # bỏ padding theo attention_mask
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum().item()

            if num_images == 0:
                cur_input_embeds = embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(image_token_indices) - 1):
                start = image_token_indices[i] + 1
                end = image_token_indices[i + 1]
                cur_input_ids_noim.append(cur_input_ids[start:end])
                cur_labels_noim.append(cur_labels[start:end])

            split_sizes = [x.shape[0] for x in cur_input_ids_noim]

            cur_input_embeds_no_im = embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds_no_im, split_sizes, dim=0)

            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])

                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype
                        )
                    )

            cur_new_input_embeds = [x.to(device=device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        position_ids = torch.zeros(
            (batch_size, max_len),
            dtype=position_ids.dtype,
            device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]

            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(
                    torch.cat((
                        torch.zeros(
                            (max_len - cur_len, embed_dim),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device
                        ),
                        cur_new_embed
                    ), dim=0)
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat((
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_len, embed_dim),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device
                        )
                    ), dim=0)
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    
    # def prepare_inputs_labels_for_multimodal(
    #     self, input_ids, position_ids, attention_mask, past_key_values, labels,
    #     images, image_sizes=None
    # ):
    #     vision_tower = self.get_vision_tower()
    #     if vision_tower is None or images is None or input_ids.shape[1] == 1:
    #         return input_ids, position_ids, attention_mask, past_key_values, None, labels

    #     # if type(images) is list or images.ndim == 5:
    #     #     if type(images) is list:
    #     #         images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
    #     #     concat_images = torch.cat([image for image in images], dim=0)
    #     #     image_features = self.encode_images(concat_images)
    #     #     split_sizes = [image.shape[0] for image in images]
    #     #     image_features = torch.split(image_features, split_sizes, dim=0)
    #     #     mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
    #     #     image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
    #     #     if mm_patch_merge_type == 'flat':
    #     #         image_features = [x.flatten(0, 1) for x in image_features]
    #     #     elif mm_patch_merge_type.startswith('spatial'):
    #     #         new_image_features = []
    #     #         for image_idx, image_feature in enumerate(image_features):
    #     #             if image_feature.shape[0] > 1:
    #     #                 base_image_feature = image_feature[0]
    #     #                 image_feature = image_feature[1:]
    #     #                 height = width = self.get_vision_tower().num_patches_per_side
    #     #                 assert height * width == base_image_feature.shape[0]
    #     #                 if image_aspect_ratio == 'anyres':
    #     #                     num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
    #     #                     image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
    #     #                 else:
    #     #                     raise NotImplementedError
    #     #                 if 'unpad' in mm_patch_merge_type:
    #     #                     image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
    #     #                     image_feature = image_feature.flatten(1, 2).flatten(2, 3)
    #     #                     image_feature = unpad_image(image_feature, image_sizes[image_idx])
    #     #                     image_feature = torch.cat((
    #     #                         image_feature,
    #     #                         self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
    #     #                     ), dim=-1)
    #     #                     image_feature = image_feature.flatten(1, 2).transpose(0, 1)
    #     #                 else:
    #     #                     image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
    #     #                     image_feature = image_feature.flatten(0, 3)
    #     #                 image_feature = torch.cat((base_image_feature, image_feature), dim=0)
    #     #             else:
    #     #                 image_feature = image_feature[0]
    #     #                 if 'unpad' in mm_patch_merge_type:
    #     #                     image_feature = torch.cat((
    #     #                         image_feature,
    #     #                         self.model.image_newline[None].to(image_feature.device)
    #     #                     ), dim=0)
    #     #             new_image_features.append(image_feature)
    #     #         image_features = new_image_features
    #     #     else:
    #     #         raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    #     # else:
    #     #    image_features = self.encode_images(images)
    #     (
    #     global_feat,                      # (B, G, D) hoặc (B, D)
    #     pet_region_embeds_per_sample,     # list[B] -> list[num_regions] -> tensor
    #     ct_region_embeds_per_sample,      # list[B] -> list[num_regions] -> tensor
    #     region_texts_per_sample,          # list[B] -> list[num_regions] -> str
    #         ) = self.encode_images(images)

    #     tokenizer = getattr(self, "tokenizer", None)
    #     if tokenizer is None:
    #         tokenizer = getattr(self.get_model(), "tokenizer", None)

    #     # seg_embeds_per_sample = None
    #     # if seg_texts is not None and tokenizer is not None:
    #     #     seg_embeds_per_sample = []
    #     #     for seg_text in seg_texts:
    #     #         seg_ids = tokenizer(
    #     #             seg_text,
    #     #             add_special_tokens=False,
    #     #             return_tensors="pt"
    #     #         ).input_ids.to(input_ids.device)

    #     #         seg_embeds = self.get_model().embed_tokens(seg_ids[0])
    #     #         seg_embeds_per_sample.append(seg_embeds)
    #     # else:
    #     #     print("codengu\n\n\n\n\n\n\n")

    #     # # TODO: image start / end is not implemented here to support pretraining.
    #     # if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
    #     #     raise NotImplementedError

    #     # # Let's just add dummy tensors if they do not exist,
    #     # # it is a headache to deal with None all the time.
    #     # # But it is not ideal, and if you have a better idea,
    #     # # please open an issue / submit a PR, thanks.
    #     # _labels = labels
    #     # _position_ids = position_ids
    #     # _attention_mask = attention_mask
    #     # if attention_mask is None:
    #     #     attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    #     # else:
    #     #     attention_mask = attention_mask.bool()
    #     # if position_ids is None:
    #     #     position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    #     # if labels is None:
    #     #     labels = torch.full_like(input_ids, IGNORE_INDEX)

    #     # # remove the padding using attention_mask -- FIXME
    #     # _input_ids = input_ids
    #     # input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    #     # labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    #     # new_input_embeds = []
    #     # new_labels = []
    #     # cur_image_idx = 0
    #     # for batch_idx, cur_input_ids in enumerate(input_ids):
    #     #     num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
    #     #     if num_images == 0:
    #     #         # cur_image_features = image_features[cur_image_idx]
    #     #         cur_image_features = image_features[cur_image_idx].unsqueeze(0)
    #     #         cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
    #     #         cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
    #     #         new_input_embeds.append(cur_input_embeds)
    #     #         new_labels.append(labels[batch_idx])
    #     #         cur_image_idx += 1
    #     #         continue

    #     #     image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
    #     #     cur_input_ids_noim = []
    #     #     cur_labels = labels[batch_idx]
    #     #     cur_labels_noim = []
    #     #     for i in range(len(image_token_indices) - 1):
    #     #         cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
    #     #         cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
    #     #     split_sizes = [x.shape[0] for x in cur_labels_noim]
    #     #     cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
    #     #     cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
    #     #     cur_new_input_embeds = []
    #     #     cur_new_labels = []

    #     #     for i in range(num_images + 1):
    #     #         cur_new_input_embeds.append(cur_input_embeds_no_im[i])
    #     #         cur_new_labels.append(cur_labels_noim[i])
    #     #         if i < num_images:
                    
    #     #             cur_image_features = image_features[cur_image_idx]
    #     #             # cur_image_features = image_features[cur_image_idx].unsqueeze(0)
    #     #             cur_image_idx += 1
    #     #             cur_new_input_embeds.append(cur_image_features)
    #     #             cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
    #     #             if seg_embeds_per_sample is not None:
    #     #                 cur_seg_embeds = seg_embeds_per_sample[batch_idx].to(self.device)
    #     #                 cur_new_input_embeds.append(cur_seg_embeds)
    #     #                 cur_new_labels.append(
    #     #                     torch.full(
    #     #                         (cur_seg_embeds.shape[0],),
    #     #                         IGNORE_INDEX,
    #     #                         device=cur_labels.device,
    #     #                         dtype=cur_labels.dtype
    #     #                     )
    #     #                 )
    #     #             else:
    #     #                 print("codengunug\n\n\n\n\n\n\n")
    #     #     # breakpoint()
    #     #     cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
                
    #     #     cur_new_input_embeds = torch.cat(cur_new_input_embeds)
    #     #     cur_new_labels = torch.cat(cur_new_labels)

    #     #     new_input_embeds.append(cur_new_input_embeds)
    #     #     new_labels.append(cur_new_labels)

    #     # # Truncate sequences to max length as image embeddings can make the sequence longer
    #     # tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
    #     # if tokenizer_model_max_length is not None:
    #     #     new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
    #     #     new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    #     # # Combine them
    #     # max_len = max(x.shape[0] for x in new_input_embeds)
    #     # batch_size = len(new_input_embeds)

    #     # new_input_embeds_padded = []
    #     # new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    #     # attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    #     # position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    #     # for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
    #     #     cur_len = cur_new_embed.shape[0]
    #     #     if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
    #     #         new_input_embeds_padded.append(torch.cat((
    #     #             torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
    #     #             cur_new_embed
    #     #         ), dim=0))
    #     #         if cur_len > 0:
    #     #             new_labels_padded[i, -cur_len:] = cur_new_labels
    #     #             attention_mask[i, -cur_len:] = True
    #     #             position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
    #     #     else:
    #     #         new_input_embeds_padded.append(torch.cat((
    #     #             cur_new_embed,
    #     #             torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
    #     #         ), dim=0))
    #     #         if cur_len > 0:
    #     #             new_labels_padded[i, :cur_len] = cur_new_labels
    #     #             attention_mask[i, :cur_len] = True
    #     #             position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    #     # new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    #     # if _labels is None:
    #     #     new_labels = None
    #     # else:
    #     #     new_labels = new_labels_padded

    #     # if _attention_mask is None:
    #     #     attention_mask = None
    #     # else:
    #     #     attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    #     # if _position_ids is None:
    #     #     position_ids = None

    #     return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False






