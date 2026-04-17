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

deepspeed --num_gpus=1 --master_port=29511 llava/train/test.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path /workdir/radish/PET-CT/PET-CT-report/pretrained_weights/llm/M3D \
    --version $PROMPT_VERSION \
    --type PET/CT \
    --image_folder /workdir/radish/PET-CT/PET-CT-report \
    --vision_tower /workdir/radish/PET-CT/PET-CT-report/ckpt/petct_emb/ctvit.89000.pt \
    --pretrain_mm_mlp_adapter /workdir/radish/PET-CT/ctvit_m3d/checkpoints/lora/checkpoint-8331/mm_projector.bin \
    --lora_path /workdir/radish/PET-CT/ctvit_m3d/checkpoints/lora/checkpoint-8331 \
    --question_file /workdir/radish/PET-CT/PET-CT-report/pretrain_data/single_turn/align_test.json \
    --temperature 0.5 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /workdir/radish/PET-CT/ctvit_m3d/checkpoints/infer/test1 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
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

