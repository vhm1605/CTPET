#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
export PYTHONPATH=/home/thaind/anonymous_project/VLMs:/home/thaind/anonymous_project/VLMs/llava/model/multimodal_encoder:$PYTHONPATH

# PROMPT_VERSION=v1
# MODEL_VERSION=llava-med-v1.5-mistral-7b
################## VICUNA ##################

################## LLaMA-2 ##################
PROMPT_VERSION="llava_llama_2"
MODEL_VERSION="llama-2-7b-chat"
# PROMPT_VERSION=plain
################## LLaMA-2 ##################

deepspeed --num_gpus=1 --master_port=29505 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path /workdir/radish/PET-CT/PET-CT-report/pretrained_weights/llm/M3D \
    --version $PROMPT_VERSION \
    --type CT \
    --data_path /workdir/radish/PET-CT/PET-CT-report/instruction/petct/ct/instruction_train_data.json \
    --eval_data_path /workdir/radish/PET-CT/PET-CT-report/instruction/petct/ct/instruction_val_data.json \
    --image_folder /workdir/radish/PET-CT/PET-CT-report \
    --vision_tower /workdir/radish/PET-CT/PET-CT-report/pretrained_weights/ct_emb/ctvit.76000.pt \
    --pretrain_mm_mlp_adapter /workdir/radish/PET-CT/ctvit_m3d/checkpoints/ct/align/checkpoint-697/mm_projector.bin \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /workdir/radish/PET-CT/ctvit_m3d/checkpoints/ct/lora_fix \
    --num_train_epochs 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to wandb
