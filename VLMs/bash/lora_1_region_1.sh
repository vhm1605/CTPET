#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
export PYTHONPATH=/home/thaind/anonymous_project/VLMs:/home/thaind/anonymous_project/VLMs/llava/model/multimodal_encoder:$PYTHONPATH

PROMPT_VERSION=v1
MODEL_VERSION=llava-med-v1.5-mistral-7b
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
# PROMPT_VERSION=plain
################## LLaMA-2 ##################

# --data_path /home/jovyan/workspace/pet_part/data_desc_conv_train.json \
# --eval_data_path /home/jovyan/workspace/pet_part/data_desc_conv_test.json \
    # --pretrain_mm_mlp_adapter /home/jovyan/workspace/pet_part/LLaVA/checkpoints/ctvit_llavamed-llava-med-v1.5-mistral-7b-pretrain-1epochs/mm_projector.bin \

# --pretrained_lora_path /home/jovyan/shared/tienhuu060102/data-petct/pretrained_weights/MultimodalFM/adaptive_cosmos_llavamed/lora/checkpoint-1740 \

deepspeed --num_gpus=1 --master_port=29503 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path /workdir/radish/PET-CT/PET-CT-report/ckpt/llava-med-v1.5-mistral-7b \
    --version $PROMPT_VERSION \
    --type PET/CT \
    --data_path /workdir/radish/PET-CT/PET-CT-report/pretrain_data/single_turn/align_train.json \
    --eval_data_path /workdir/radish/PET-CT/PET-CT-report/pretrain_data/single_turn/align_val.json \
    --image_folder /workdir/radish/PET-CT/PET-CT-report \
    --vision_tower /workdir/radish/PET-CT/PET-CT-report/ckpt/petct_emb/ctvit.89000.pt \
    --pretrain_mm_mlp_adapter /workdir/radish/PET-CT/ctvit_llavamed/checkpoints/lora_region/checkpoint-11144/mm_projector.bin \
    --pretrained_lora_path /workdir/radish/PET-CT/ctvit_llavamed/checkpoints/lora_region/checkpoint-11144 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /workdir/radish/PET-CT/ctvit_llavamed/checkpoints/lora_region_1 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to wandb
