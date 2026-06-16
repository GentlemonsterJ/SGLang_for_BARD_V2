from transformers import CONFIG_MAPPING, PretrainedConfig


class DstarVLVisionConfig(PretrainedConfig):
    model_type = "dstar_vl"
    base_config_key = "vision_config"
    
    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[8, 16, 24],
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = deepstack_visual_indexes


class DstarVLTextConfig(PretrainedConfig):
    model_type = "dstar_vl_text"
    base_config_key = "text_config"
    
    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=128000,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=5000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class DstarVLConfig(PretrainedConfig):
    model_type = "dstar_vl"
    sub_configs = {
        "vision_config": DstarVLVisionConfig,
        "text_config": DstarVLTextConfig,
    }

    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["DstarVLForConditonalGeneration"]

        if isinstance(vision_config, dict):
            vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            vision_config = self.sub_configs["vision_config"]()
        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            text_config = self.sub_configs["text_config"]()
        self.text_config = text_config

        # Expose text-config fields at the top level because the language model
        # consumes DstarVLConfig directly, not config.text_config.
        self.vocab_size = text_config.vocab_size
        self.max_position_embeddings = text_config.max_position_embeddings
        self.hidden_size = text_config.hidden_size
        self.intermediate_size = text_config.intermediate_size
        self.num_hidden_layers = text_config.num_hidden_layers
        self.num_attention_heads = text_config.num_attention_heads
        self.num_key_value_heads = text_config.num_key_value_heads
        self.head_dim = text_config.head_dim
        self.hidden_act = text_config.hidden_act
        self.initializer_range = text_config.initializer_range
        self.rms_norm_eps = text_config.rms_norm_eps
        self.use_cache = text_config.use_cache
        self.rope_theta = text_config.rope_theta
        self.rope_scaling = text_config.rope_scaling
        self.attention_bias = text_config.attention_bias
        self.attention_dropout = text_config.attention_dropout

        for attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
            value = getattr(text_config, attr, None)
            if value is not None:
                setattr(self, attr, value)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(
            **kwargs,
            tie_word_embeddings=getattr(
                text_config, "tie_word_embeddings", tie_word_embeddings
            ),
        )


try:
    CONFIG_MAPPING.register("dstar_vl", DstarVLConfig)
except Exception:
    # Already registered or registration failed; fall back to direct assignment.
    CONFIG_MAPPING._extra_content["dstar_vl"] = DstarVLConfig
