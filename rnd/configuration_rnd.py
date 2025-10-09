# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""
RND1 Model Configuration.

This module defines the configuration class for RND1 models,
extending Qwen3MoeConfig with RND1-specific parameters.
"""

from typing import Optional
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


class RND1Config(Qwen3MoeConfig):
    """
    Configuration class for RND1 models.

    This configuration extends Qwen3MoeConfig with additional parameters
    specific to the RND1 (Radical Numerics Diffusion v1) architecture.

    Args:
        moe_backend: Backend for MoE computation ("hf", "flashinfer", or "sglang")
        num_diffusion_steps: Default number of diffusion steps for generation
        mask_token_id: Token ID used for masking (default: 151669 for Qwen)
        **kwargs: Additional arguments passed to Qwen3MoeConfig
    """

    model_type = "rnd1"

    def __init__(
        self,
        moe_backend: str = "hf",
        num_diffusion_steps: int = 256,
        mask_token_id: int = 151669,  # Default for Qwen-based RND1 models
        use_cache: bool = False,
        **kwargs,
    ):
        # Force non-causal and no caching for RND1
        kwargs['use_cache'] = False
        kwargs['is_causal'] = False
        super().__init__(**kwargs)
        
        # `head_dim` needs to be 128 for Qwen3MoE
        # need to ensure that the config has this attr if directly passing config to RND1LM at instantiation
        if not hasattr(self, "head_dim"):
            self.head_dim = 128

        # Note that in transformers 4.57.0 there is an error in the config
        # num_hidden_layers is defaulted to 24
        self.num_hidden_layers = 48

        # RND1-specific parameters
        self.moe_backend = moe_backend
        self.num_diffusion_steps = num_diffusion_steps
        self.mask_token_id = mask_token_id

        # Ensure bidirectional attention and no caching
        self.is_causal = False
        self.use_cache = False

    def to_dict(self):
        """
        Serializes configuration to dictionary with auto_map for Hub.

        The auto_map ensures that when users load from HuggingFace Hub,
        the correct custom classes are automatically resolved.
        """
        data = super().to_dict()
        data.setdefault("auto_map", {
            "AutoConfig": "configuration_rnd.RND1Config",
            "AutoModel": "modeling_rnd.RND1Model",
            "AutoModelForMaskedLM": "modeling_rnd.RND1LM",
        })
        return data