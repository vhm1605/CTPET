import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import _is_peft_model
from transformers.trainer import *
from typing import List, Optional
from llava.constants import IMAGE_TOKEN_INDEX
import torch.nn.functional as F
from scipy.ndimage import label, binary_closing

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):

        
        # if getattr(self.args, 'tune_mm_mlp_adapter', False):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        # Only save Adapter
        keys_to_match = ['mm_projector', 'vision_resampler']
        if getattr(self.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

        if self.args.local_rank == 0 or self.args.local_rank == -1:
            self.model.config.save_pretrained(output_dir)
            torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

        if getattr(self.args, 'lora_enable', False):
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # if getattr(self.args, 'tune_mm_mlp_adapter', False):
        #     pass
        # else:
        #     super(LLaVATrainer, self)._save(output_dir, state_dict)
        if getattr(self.args, 'lora_enable', False):
            super(LLaVATrainer, self)._save(output_dir, state_dict)
        else:
            pass
        
    def clean_mask(self, mask, min_size=50):
        """
        Loại bỏ các cluster uptake nhỏ hơn min_size voxel.
        mask: (B, D, H, W) binary
        """
        mask_np = mask.cpu().numpy()
        cleaned_np = np.zeros_like(mask_np)
        for b in range(mask_np.shape[0]):
            labeled, num_features = label(mask_np[b])
            for i in range(1, num_features + 1):
                if (labeled == i).sum() >= min_size:
                    cleaned_np[b][labeled == i] = 1
        return torch.tensor(cleaned_np, device=mask.device, dtype=mask.dtype)

    def close_mask(self, mask, iterations=2):
        """
        Kết nối vùng uptake bị rời rạc, fill các lỗ hổng nhỏ.
        mask: (B, D, H, W) binary
        """
        mask_np = mask.cpu().numpy()
        closed_np = np.zeros_like(mask_np)
        for b in range(mask_np.shape[0]):
            closed_np[b] = binary_closing(mask_np[b], iterations=iterations)
        return torch.tensor(closed_np, device=mask.device, dtype=mask.dtype)

    def soft_dilate_mask(self, mask, radius=1):
        """
        Giãn mask bằng conv3d trên GPU.
        mask: (B, 1, D, H, W) binary
        """
        kernel_size = 2 * radius + 1
        kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=mask.device)
        dilated = F.conv3d(mask, kernel, padding=radius)
        dilated = (dilated > 0).float()
        return dilated

    def adaptive_dilate_mask(self, mask, target_mean=0.12, max_iter=5):
        """
        Giãn mask đến khi đạt mean ~ target_mean.
        mask: (B, 1, D, H, W)
        """
        out = mask.clone()
        iter_count = 0
        while out.mean().item() < target_mean and iter_count < max_iter:
            out = self.soft_dilate_mask(out, radius=1)
            iter_count += 1
        return out

    def process_pet_mask(self, pet_tensor, threshold=0.5, min_size=50, close_iter=2, target_mean=0.12):
        """
        Input:
            pet_tensor: (B, 1, D, H, W), normalized 0..1
        Output:
            mask_final: (B, 1, D, H, W) binary mask đã xử lý
        """
        B, C, D, H, W = pet_tensor.shape
        
        # Threshold ban đầu
        mask = (pet_tensor > threshold).float().view(B, D, H, W)

        # Clean noise nhỏ
        mask = self.clean_mask(mask, min_size=min_size)

        # Morphological closing
        mask = self.close_mask(mask, iterations=close_iter)

        # Add channel để conv3d
        mask = mask.unsqueeze(1)

        # Adaptive dilation
        mask_final = self.adaptive_dilate_mask(mask, target_mean=target_mean)

        return mask_final
        
    # def compute_alignment_loss(self, attn, pet_mask, eps=1e-6):
    #     """
    #     attn_reshaped: (B, D * H * W), normalized attention map từ text → visual
    #     pet_mask: (B, D * H * W), binary mask (0/1) từ PET
    #     return: scalar loss
    #     """
        
    #     total_attn = attn.sum(dim=1) + eps
    #     region_attn = (attn * pet_mask).sum(dim=1)
    #     alignment = 1 - (region_attn / total_attn)
    #     loss = (alignment ** 2).mean()

    #     return loss
    
    def compute_alignment_loss(self, attn, pet_mask, alpha=0.3, beta=0.7, gamma=2.0, eps=1e-6):
        """
        Focal Tversky Loss cho alignment giữa attention và PET mask.

        attn: (B, D*H*W) attention map từ text → visual
        pet_mask: (B, D*H*W) binary mask (0/1) lesions
        alpha: trọng số FP
        beta: trọng số FN (ưu tiên cao nếu lesions nhỏ)
        gamma: exponent >1 để tập trung vào hard cases
        """
        # Normalize attn về [0,1]
        attn = attn - attn.min(dim=1, keepdim=True)[0]
        attn = attn / (attn.max(dim=1, keepdim=True)[0] + eps)

        # TP, FP, FN
        TP = (attn * pet_mask).sum(dim=1)
        FP = (attn * (1 - pet_mask)).sum(dim=1)
        FN = ((1 - attn) * pet_mask).sum(dim=1)

        # Tversky index
        tversky = (TP + eps) / (TP + alpha * FP + beta * FN + eps)

        # Focal Tversky Loss
        loss = (1 - tversky) ** gamma
        return loss.mean()
    
    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, llm_loss, align_loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            elif is_torch_hpu_available():
                logger.warning(
                    "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                )
            else:
                torch.cuda.empty_cache()
                
        self.log({"llm_loss": llm_loss.item(), "align_loss": align_loss.item()})

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()
        
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        total_loss, llm_loss, align_loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                        loss = total_loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        
        outputs = model(**inputs, output_attentions=True)
        print(inputs.keys())
        pet_tensor = inputs["images"]["PET"]
        ct_tensor = inputs["images"]["CT"]

        model_ = model.module if hasattr(model, "module") else model
        vis_feat = model_.get_vision_tower()(pet_tensor, ct_tensor)

        B, D_, H_, W_, C = vis_feat.shape
        D_conv, H_conv, W_conv = int(D_ / 2), int(H_ / 3), int(W_ / 3)
        N = int(D_conv * H_conv * W_conv)

        # Get LLM attention to visual tokens
        attn_tensor = torch.stack(outputs.attentions)  # (L, B, H, T, T)
        L, B, H, T, _ = attn_tensor.shape
        attn = attn_tensor.permute(1, 0, 2, 3, 4)  # (B, L, H, T, T)
        attn = attn.reshape(B, L * H, T, T)        # (B, L*H, T, T)

        # Tìm index của <image> token cho mỗi sample trong batch
        input_ids = inputs["input_ids"]
        img_token_idx = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)
        img_token_idx = img_token_idx[:, 1]  # (B,)
        assert img_token_idx.shape[0] == B

        # Tạo mask để extract visual attention (vectorized trick)
        arangeN = torch.arange(N, device=attn.device).view(1, N)  # (1, N)
        img_token_idx_expand = img_token_idx.view(B, 1) + arangeN   # (B, N)
        img_token_idx_expand = img_token_idx_expand.unsqueeze(1).expand(B, L * H, N)  # (B, L*H, N)

        # Gather attn[:, :, -1, img_token_idx:img_token_idx+N] for all samples
        attn_vis = torch.gather(attn[:, :, -1, :], 2, img_token_idx_expand)  # (B, L*H, N)
        attn_txt = attn[:, :, -1, :]  # (B, L*H, T)

        r_lh = attn_vis.sum(dim=-1) / (attn_txt.sum(dim=-1) + 1e-6)  # (B, L*H)

        # Lấy top-R heads theo r_lh
        top_r = 64
        _, top_r_idx = torch.topk(r_lh, top_r, dim=-1)  # (B, R)

        # Tạo mask attention weight cho top-R
        attn_top = torch.gather(attn_vis, 1, top_r_idx.unsqueeze(-1).expand(-1, -1, N))  # (B, R, N)
        attn_weights = attn_top.mean(dim=1)  # (B, N)

        # Compute voxel-level attention
        attn_voxel_3d = attn_weights.view(B, 1, D_conv, H_conv, W_conv)

        _, _, D, H, W = pet_tensor.shape
        # Interpolate lên đúng shape PET gốc
        attn_voxel_up = F.interpolate(
            attn_voxel_3d, size=(D, H, W), mode='trilinear', align_corners=False
        )  # (B, 1, D, H, W)

        # Flatten để tính alignment loss
        attn_voxel_up = attn_voxel_up.view(B, D * H * W)
        # attn_voxel_up = attn_voxel_up / (attn_voxel_up.sum(dim=1, keepdim=True) + 1e-6)

        pet_mask = self.process_pet_mask(pet_tensor)  # (B, 1, D, H, W)

        # Flatten
        pet_mask = pet_mask.view(B, D * H * W)
        align_loss = self.compute_alignment_loss(attn_voxel_up, pet_mask)

        print("attn_voxel stats:", attn_voxel_up.mean().item(), attn_voxel_up.min().item(), attn_voxel_up.max().item())
        print("pet_mask stats:", pet_mask.mean().item(), pet_mask.min().item(), pet_mask.max().item())
        
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        total_loss = loss + 0.5 * align_loss

        if return_outputs:
            return total_loss, loss, align_loss, outputs 
        else:
            return total_loss, loss, align_loss
        
    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     """
    #     How the loss is computed by Trainer. By default, all models return the loss in the first element.

    #     Subclass and override for custom behavior.
    #     """
    #     if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
    #         labels = inputs.pop("labels")
    #     else:
    #         labels = None
    #     if self.model_accepts_loss_kwargs:
    #         loss_kwargs = {}
    #         if num_items_in_batch is not None:
    #             loss_kwargs["num_items_in_batch"] = num_items_in_batch
    #         inputs = {**inputs, **loss_kwargs}
    #     outputs = model(**inputs, output_attentions=True)
    #     pet_tensor = inputs["images"]["PET"]
    #     ct_tensor = inputs["images"]["CT"]
        
    #     model_ = model.module if hasattr(model, "module") else model
    #     vis_feat = model_.get_vision_tower()(pet_tensor, ct_tensor)
        
    #     B, D_, H_, W_, C = vis_feat.shape

    #     # Run projector with attn
    #     _, attn_latent2voxel = model_.get_projector()(vis_feat, return_attn=True)  # attn: (B, 1, 64, D*H*W)
    #     attn_latent2voxel = attn_latent2voxel[:, 0]  # (B, 64, D*H*W)

    #     # Get LLM attention to visual tokens
    #     attn_tensor = torch.stack(outputs.attentions)  # (L, B, H, T, T)
    #     L, B, H, T, _ = attn_tensor.shape
    #     attn = attn_tensor.permute(1, 0, 2, 3, 4)  # (B, L, H, T, T)
    #     attn = attn.reshape(B, L * H, T, T)        # (B, L*H, T, T)

    #     # Tìm index của <image> token cho mỗi sample trong batch
    #     input_ids = inputs["input_ids"]
    #     img_token_idx = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)
    #     img_token_idx = img_token_idx[:, 1]  # (B,)
    #     assert img_token_idx.shape[0] == B

    #     # Tạo mask để extract visual attention (vectorized trick)
    #     arange64 = torch.arange(64, device=attn.device).view(1, 64)  # (1, 64)
    #     img_token_idx_expand = img_token_idx.view(B, 1) + arange64   # (B, 64)
    #     img_token_idx_expand = img_token_idx_expand.unsqueeze(1).expand(B, L * H, 64)  # (B, L*H, 64)

    #     # Gather attn[:, :, -1, img_token_idx:img_token_idx+64] for all samples
    #     attn_vis = torch.gather(attn[:, :, -1, :], 2, img_token_idx_expand)  # (B, L*H, 64)
    #     attn_txt = attn[:, :, -1, :]  # (B, L*H, T)

    #     r_lh = attn_vis.sum(dim=-1) / (attn_txt.sum(dim=-1) + 1e-6)  # (B, L*H)
    #     weighted_loss = ((1.0 - r_lh.mean(dim=-1)) ** 2).mean()

    #     # Lấy top-R heads theo r_lh
    #     top_r = 128
    #     _, top_r_idx = torch.topk(r_lh, top_r, dim=-1)  # (B, R)

    #     # Tạo mask attention weight cho top-R
    #     attn_top = torch.gather(attn_vis, 1, top_r_idx.unsqueeze(-1).expand(-1, -1, 64))  # (B, R, 64)
    #     attn_weights = attn_top.mean(dim=1)  # (B, 64)

    #     # Compute voxel-level attention
    #     # attn_voxel = attn_latent2voxel.mean(dim=1)
    #     attn_voxel = torch.bmm(attn_weights.unsqueeze(1), attn_latent2voxel)  # (B, 1, D*H*W)
    #     attn_voxel_3d = attn_voxel.view(B, 1, D_, H_, W_)

    #     _, _, D, H, W = pet_tensor.shape
    #     # Interpolate lên đúng shape PET gốc
    #     attn_voxel_up = F.interpolate(
    #         attn_voxel_3d, size=(D, H, W), mode='trilinear', align_corners=False
    #     )  # (B, 1, D, H, W)

    #     # Flatten để tính alignment loss
    #     attn_voxel_up = attn_voxel_up.view(B, D * H * W)
    #     attn_voxel_up = attn_voxel_up / (attn_voxel_up.sum(dim=1, keepdim=True) + 1e-6)

    #     pet_mask = (pet_tensor > 0.2).float()  # (B, 1, D, H, W)

    #     # Flatten
    #     pet_mask = pet_mask.view(B, D * H * W)

    #     align_loss = self.compute_alignment_loss(attn_voxel_up, pet_mask)

    #     print("attn_voxel stats:", attn_voxel_up.mean().item(), attn_voxel_up.min().item(), attn_voxel_up.max().item())
    #     print("pet_mask stats:", pet_mask.mean().item(), pet_mask.min().item(), pet_mask.max().item())
        
    #     # Save past state if it exists
    #     # TODO: this needs to be fixed and made cleaner later.
    #     if self.args.past_index >= 0:
    #         self._past = outputs[self.args.past_index]

    #     if labels is not None:
    #         unwrapped_model = self.accelerator.unwrap_model(model)
    #         if _is_peft_model(unwrapped_model):
    #             model_name = unwrapped_model.base_model.model._get_name()
    #         else:
    #             model_name = unwrapped_model._get_name()
    #         # User-defined compute_loss function
    #         if self.compute_loss_func is not None:
    #             loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
    #         elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
    #             loss = self.label_smoother(outputs, labels, shift_labels=True)
    #         else:
    #             loss = self.label_smoother(outputs, labels)
    #     else:
    #         if isinstance(outputs, dict) and "loss" not in outputs:
    #             raise ValueError(
    #                 "The model did not return a loss from the inputs, only the following keys: "
    #                 f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
    #             )
    #         # We don't use .loss here since the model may return tuples instead of ModelOutput.
    #         loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    #     if (
    #         self.args.average_tokens_across_devices
    #         and (self.model_accepts_loss_kwargs or self.compute_loss_func)
    #         and num_items_in_batch is not None
    #     ):
    #         loss *= self.accelerator.num_processes

    #     total_loss = loss + 0.05 * align_loss + 0.3 * weighted_loss

    #     if return_outputs:
    #         return total_loss, loss, align_loss, weighted_loss, outputs 
    #     else:
    #         return total_loss, loss, align_loss, weighted_loss
