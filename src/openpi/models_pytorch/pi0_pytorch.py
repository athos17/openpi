import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def masked_token_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    losses = losses.reshape(targets.shape)
    mask = mask.to(dtype=losses.dtype, device=losses.device)
    denom = torch.clamp(mask.sum(dim=-1), min=1.0)
    return (losses * mask).sum(dim=-1) / denom


def masked_lm_head_cross_entropy(
    lm_head: nn.Module,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sample token CE while projecting only supervised positions."""
    batch_size = targets.shape[0]
    mask = mask.to(device=hidden.device, dtype=torch.bool)
    denom = torch.clamp(mask.sum(dim=-1), min=1)
    if not torch.any(mask):
        return torch.zeros(batch_size, device=hidden.device, dtype=torch.float32)

    flat_mask = mask.reshape(-1)
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)
    sample_ids = torch.arange(batch_size, device=hidden.device)[:, None].expand_as(mask).reshape(-1)

    selected_logits = lm_head(flat_hidden[flat_mask]).to(dtype=torch.float32)
    selected_losses = F.cross_entropy(selected_logits, flat_targets[flat_mask], reduction="none")
    losses = torch.zeros(batch_size, device=hidden.device, dtype=torch.float32)
    losses.scatter_add_(0, sample_ids[flat_mask], selected_losses)
    return losses / denom.to(dtype=torch.float32)


def mask_action_tokens_for_flow(
    tokenized_prompt: torch.Tensor,
    tokenized_prompt_mask: torch.Tensor,
    action_region_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if action_region_mask is None:
        return tokenized_prompt, tokenized_prompt_mask
    return (
        torch.where(action_region_mask, torch.zeros_like(tokenized_prompt), tokenized_prompt),
        torch.where(action_region_mask, torch.zeros_like(tokenized_prompt_mask, dtype=torch.bool), tokenized_prompt_mask),
    )


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        for module in (
            self.paligemma_with_expert.paligemma.language_model,
            self.paligemma_with_expert.paligemma.vision_tower,
            self.paligemma_with_expert.gemma_expert.model,
        ):
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                module.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        for module in (
            self.paligemma_with_expert.paligemma.language_model,
            self.paligemma_with_expert.paligemma.vision_tower,
            self.paligemma_with_expert.gemma_expert.model,
        ):
            if hasattr(module, "gradient_checkpointing_disable"):
                module.gradient_checkpointing_disable()
            else:
                module.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks, dtype=None):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        attention_bias = torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)
        if dtype is not None:
            attention_bias = attention_bias.to(dtype=dtype)
        return attention_bias

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def _compute_token_losses_from_lang_hidden(
        self,
        lang_hidden: torch.Tensor,
        lang_tokens: torch.Tensor,
        subtask_region_mask: torch.Tensor | None,
        action_region_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = lang_tokens[:, 1:].to(device=lang_hidden.device, dtype=torch.long)

        if subtask_region_mask is None:
            subtask_mask = torch.zeros_like(targets, dtype=torch.bool)
        else:
            subtask_mask = subtask_region_mask[:, 1:].to(device=lang_hidden.device, dtype=torch.bool)
        if action_region_mask is None:
            action_mask = torch.zeros_like(targets, dtype=torch.bool)
        else:
            action_mask = action_region_mask[:, 1:].to(device=lang_hidden.device, dtype=torch.bool)

        return (
            masked_lm_head_cross_entropy(
                self.paligemma_with_expert.paligemma.lm_head,
                lang_hidden,
                targets,
                subtask_mask,
            ),
            masked_lm_head_cross_entropy(
                self.paligemma_with_expert.paligemma.lm_head,
                lang_hidden,
                targets,
                action_mask,
            ),
        )

    def compute_token_losses(self, observation) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-sample subtask and FAST action-token cross-entropy losses."""
        processed = _preprocessing.preprocess_observation_pytorch(observation, train=True)
        images = list(processed.images.values())
        img_masks = list(processed.image_masks.values())
        lang_tokens = processed.tokenized_prompt
        lang_masks = processed.tokenized_prompt_mask

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        if processed.token_ar_mask is not None:
            prefix_att_masks = prefix_att_masks.clone()
            prefix_att_masks[:, -lang_tokens.shape[1] :] = processed.token_ar_mask.to(
                device=prefix_att_masks.device,
                dtype=prefix_att_masks.dtype,
            )
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks, dtype=prefix_embs.dtype)

        def token_forward_func(prefix_embs, att_2d_masks_4d, position_ids):
            (prefix_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
                adarms_cond=[None, None],
            )
            return prefix_out

        prefix_out = self._apply_checkpoint(token_forward_func, prefix_embs, att_2d_masks_4d, position_ids)

        lang_hidden = prefix_out[:, -lang_tokens.shape[1] :][:, :-1]
        return self._compute_token_losses_from_lang_hidden(
            lang_hidden,
            lang_tokens,
            processed.subtask_region_mask,
            processed.action_region_mask,
        )

    def compute_flow_loss(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks, dtype=prefix_embs.dtype)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        self._last_flow_token_context = (
            prefix_out[:, -lang_tokens.shape[1] :][:, :-1],
            lang_tokens,
            getattr(observation, "subtask_region_mask", None),
            getattr(observation, "action_region_mask", None),
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    def forward(self, observation, actions, noise=None, time=None):
        """Do a full training forward pass and compute enabled weighted objectives."""
        total_loss = torch.zeros(actions.shape[0], device=actions.device, dtype=torch.float32)
        token_losses_needed = self.config.subtask_loss_weight > 0 or self.config.fast_token_loss_weight > 0
        subtask_loss = torch.zeros(actions.shape[0], device=actions.device, dtype=torch.float32)
        fast_token_loss = torch.zeros(actions.shape[0], device=actions.device, dtype=torch.float32)

        if self.config.flow_matching_loss_weight > 0:
            flow_observation = observation
            if self.config.fast_token_loss_weight > 0 and getattr(observation, "action_region_mask", None) is not None:
                flow_tokens, flow_token_mask = mask_action_tokens_for_flow(
                    observation.tokenized_prompt,
                    observation.tokenized_prompt_mask,
                    observation.action_region_mask,
                )
                if hasattr(observation, "replace"):
                    flow_observation = observation.replace(
                        tokenized_prompt=flow_tokens,
                        tokenized_prompt_mask=flow_token_mask,
                    )
                else:
                    flow_observation = type(observation)(
                        images=observation.images,
                        image_masks=observation.image_masks,
                        state=observation.state,
                        tokenized_prompt=flow_tokens,
                        tokenized_prompt_mask=flow_token_mask,
                        token_ar_mask=getattr(observation, "token_ar_mask", None),
                        token_loss_mask=getattr(observation, "token_loss_mask", None),
                        subtask_region_mask=getattr(observation, "subtask_region_mask", None),
                        action_region_mask=getattr(observation, "action_region_mask", None),
                    )
            self._last_flow_token_context = None
            flow_loss = self.compute_flow_loss(flow_observation, actions, noise=noise, time=time).mean(dim=(-1, -2))
        else:
            flow_loss = torch.zeros(actions.shape[0], device=actions.device, dtype=torch.float32)

        if token_losses_needed:
            flow_token_context = getattr(self, "_last_flow_token_context", None)
            can_reuse_flow_prefix = (
                flow_token_context is not None
                and self.config.flow_matching_loss_weight > 0
                and self.config.fast_token_loss_weight == 0
            )
            if can_reuse_flow_prefix:
                subtask_loss, fast_token_loss = self._compute_token_losses_from_lang_hidden(*flow_token_context)
            else:
                subtask_loss, fast_token_loss = self.compute_token_losses(observation)

        total_loss = total_loss + self.config.subtask_loss_weight * subtask_loss
        total_loss = total_loss + self.config.fast_token_loss_weight * fast_token_loss
        total_loss = total_loss + self.config.flow_matching_loss_weight * flow_loss

        return {
            "loss": total_loss.mean(),
            "flow_loss": flow_loss.mean(),
            "subtask_loss": subtask_loss.mean(),
            "fast_token_loss": fast_token_loss.mean(),
        }

    @torch.no_grad()
    def sample_low_level_task(
        self,
        device,
        observation,
        max_decoding_steps: int = 200,
        paligemma_eos_token: int = 1,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, object, torch.Tensor, torch.Tensor]:
        """Autoregressively decode low-level subtask tokens from a high-level WUJI prefix."""
        images, img_masks, lang_tokens, lang_masks, _ = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        token_ar_mask = getattr(observation, "token_ar_mask", None)
        if token_ar_mask is not None:
            prefix_att_masks = prefix_att_masks.clone()
            token_ar_mask = torch.as_tensor(token_ar_mask, device=prefix_att_masks.device, dtype=prefix_att_masks.dtype)
            prefix_att_masks[:, -lang_tokens.shape[1] :] = token_ar_mask
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks, dtype=prefix_embs.dtype)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )

        batch_size = prefix_pad_masks.shape[0]
        prefix_lengths = prefix_pad_masks.sum(dim=-1)
        last_indices = torch.clamp(prefix_lengths - 1, min=0).to(dtype=torch.long)
        last_hidden = prefix_out[torch.arange(batch_size, device=prefix_out.device), last_indices][:, None]
        logits = self.paligemma_with_expert.paligemma.lm_head(last_hidden).to(dtype=torch.float32)
        log_probs = torch.log_softmax(logits, dim=-1)

        output_tokens = []
        generated_masks = []
        generated_att_masks = []
        all_eos = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for step in range(max_decoding_steps):
            if temperature > 0.0:
                token = torch.distributions.Categorical(logits=log_probs[:, -1] / temperature).sample()
            else:
                token = torch.argmax(log_probs[:, -1], dim=-1)
            output_tokens.append(token)
            generated_masks.append(torch.ones(batch_size, dtype=torch.bool, device=device))
            generated_att_masks.append(torch.ones(batch_size, dtype=prefix_att_masks.dtype, device=device))

            token_emb = self.paligemma_with_expert.embed_language_tokens(token[:, None])
            token_emb = token_emb * math.sqrt(token_emb.shape[-1])
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                token_emb = token_emb.to(dtype=torch.bfloat16)

            generated_pad_masks = torch.stack(generated_masks, dim=1)
            current_prefix_mask = torch.cat([prefix_pad_masks, generated_pad_masks], dim=1)
            query_attn_mask = current_prefix_mask[:, None, :]
            query_attn_mask_4d = self._prepare_attention_masks_4d(query_attn_mask, dtype=token_emb.dtype)
            position_ids = (prefix_lengths + step)[:, None].to(dtype=torch.long, device=device)

            (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=query_attn_mask_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[token_emb, None],
                use_cache=True,
                adarms_cond=[None, None],
            )
            logits = self.paligemma_with_expert.paligemma.lm_head(prefix_out[:, -1:]).to(dtype=torch.float32)
            log_probs = torch.log_softmax(logits, dim=-1)

            all_eos = all_eos | (token == paligemma_eos_token)
            if torch.all(all_eos):
                break

        if output_tokens:
            output_tokens_tensor = torch.stack(output_tokens, dim=1)
            generated_pad_masks = torch.stack(generated_masks, dim=1)
            generated_att_masks_tensor = torch.stack(generated_att_masks, dim=1)
        else:
            output_tokens_tensor = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            generated_pad_masks = torch.empty((batch_size, 0), dtype=torch.bool, device=device)
            generated_att_masks_tensor = torch.empty((batch_size, 0), dtype=prefix_att_masks.dtype, device=device)

        prefix_pad_masks = torch.cat([prefix_pad_masks, generated_pad_masks], dim=1)
        prefix_att_masks = torch.cat([prefix_att_masks, generated_att_masks_tensor], dim=1)
        return output_tokens_tensor, past_key_values, prefix_pad_masks, prefix_att_masks

    @torch.no_grad()
    def sample_actions_with_subtask(
        self,
        device,
        observation,
        noise=None,
        num_steps=10,
        max_subtask_decoding_steps: int = 200,
        subtask_temperature: float = 0.0,
    ) -> Tensor:
        """Generate a low-level subtask prefix, then denoise continuous actions from that prefix."""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        _, _, _, _, state = self._preprocess_observation(observation, train=False)
        output_tokens, past_key_values, prefix_pad_masks, _ = self.sample_low_level_task(
            device,
            observation,
            max_decoding_steps=max_subtask_decoding_steps,
            paligemma_eos_token=1,
            temperature=subtask_temperature,
        )
        del output_tokens

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks, dtype=prefix_embs.dtype)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks, dtype=suffix_embs.dtype)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
