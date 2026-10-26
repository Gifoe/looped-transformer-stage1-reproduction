"""Runtime loop patch for Qwen-style decoder-only Transformers.

The implementation follows the official reproduction recipe but keeps the
numerical loop strategies in ``tflt_repro.strategies`` so they can be tested
without a Hugging Face model.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch

from ..strategies import anchored_euler, iterate_map


@dataclass(frozen=True)
class LoopConfig:
    loop_indices: list[int]
    K: int = 6
    strategy: str = "block_anchored"
    loop_mode: str = "block"
    cache_strategy: str = "first"
    decode_mode: str = "bypass"
    decode_first_n: int = 0
    anchor_beta: float = 0.5
    ema_alpha: float = 0.5
    momentum_beta: float = 0.3
    anderson_m: int = 2
    anderson_beta: float = 1.0
    per_step_alphas: list[float] | None = None
    halt_tau: float | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LoopConfig":
        values = dict(data)
        if "indices" in values and "loop_indices" not in values:
            values["loop_indices"] = values.pop("indices")
        return cls(**values)


def patch_qwen_loop(model: Any, config: LoopConfig | Mapping[str, Any]) -> Any:
    """Patch a Qwen/Qwen-MoE causal LM or inner model in-place."""

    cfg = config if isinstance(config, LoopConfig) else LoopConfig.from_mapping(config)
    _validate_config(cfg)
    inner = model.model if hasattr(model, "model") else model
    if not hasattr(inner, "_tflt_original_forward"):
        inner._tflt_original_forward = inner.forward

    inner._tflt_loop_indices = list(cfg.loop_indices)
    inner._tflt_K = int(cfg.K)
    inner._tflt_strategy = cfg.strategy
    inner._tflt_loop_mode = cfg.loop_mode
    inner._tflt_cache_strategy = cfg.cache_strategy
    inner._tflt_decode_mode = cfg.decode_mode
    inner._tflt_decode_first_n = int(cfg.decode_first_n)
    inner._tflt_decode_count = 0
    inner._tflt_anchor_beta = float(cfg.anchor_beta)
    inner._tflt_ema_alpha = float(cfg.ema_alpha)
    inner._tflt_momentum_beta = float(cfg.momentum_beta)
    inner._tflt_anderson_m = int(cfg.anderson_m)
    inner._tflt_anderson_beta = float(cfg.anderson_beta)
    inner._tflt_per_step_alphas = (
        [float(alpha) for alpha in cfg.per_step_alphas] if cfg.per_step_alphas else None
    )
    inner._tflt_halt_tau = float(cfg.halt_tau) if cfg.halt_tau is not None else None
    inner.forward = MethodType(_looped_forward, inner)
    return model


def unpatch_qwen_loop(model: Any) -> Any:
    """Restore the original inner-model forward if this module patched it."""

    inner = model.model if hasattr(model, "model") else model
    if hasattr(inner, "_tflt_original_forward"):
        inner.forward = inner._tflt_original_forward
        delattr(inner, "_tflt_original_forward")
    return model


def describe_loop(model: Any) -> str:
    inner = model.model if hasattr(model, "model") else model
    if not hasattr(inner, "_tflt_loop_indices"):
        return "no loop patch"
    parts = [
        f"indices={inner._tflt_loop_indices}",
        f"K={inner._tflt_K}",
        f"strategy={inner._tflt_strategy}",
        f"mode={inner._tflt_loop_mode}",
        f"cache={inner._tflt_cache_strategy}",
        f"decode={inner._tflt_decode_mode}",
    ]
    if inner._tflt_strategy in {"block_anchored", "per_layer_anchored"}:
        parts.append(f"anchor_beta={inner._tflt_anchor_beta}")
    if inner._tflt_strategy in {"ema", "heavy_ball"}:
        parts.append(f"alpha={inner._tflt_ema_alpha}")
    return " ".join(parts)


def _looped_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: Any = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Any = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
):
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("pass exactly one of input_ids or inputs_embeds")

    use_cache = bool(getattr(self.config, "use_cache", False) if use_cache is None else use_cache)
    return_dict = bool(getattr(self.config, "use_return_dict", True) if return_dict is None else return_dict)
    if use_cache and past_key_values is None:
        past_key_values = _new_dynamic_cache(self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen = _cache_seq_length(past_key_values) if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen,
            past_seen + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    hidden_states = inputs_embeds
    position_embeddings = _rotary_embeddings(self, hidden_states, position_ids)
    masks = _build_causal_masks(
        self,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    layers = self.layers
    n_layers = len(layers)
    loop_start, loop_end = _loop_bounds(self._tflt_loop_indices)
    if loop_start < 0 or loop_end > n_layers:
        raise ValueError(f"loop indices {self._tflt_loop_indices} outside {n_layers} layers")

    seq_len = inputs_embeds.shape[1]
    is_incremental = past_key_values is not None and seq_len == 1
    if is_incremental:
        self._tflt_decode_count = getattr(self, "_tflt_decode_count", 0) + 1
    else:
        self._tflt_decode_count = 0

    do_loop = _should_loop(
        is_incremental=is_incremental,
        decode_mode=self._tflt_decode_mode,
        decode_count=self._tflt_decode_count,
        decode_first_n=self._tflt_decode_first_n,
    )

    def call_layer(layer_idx: int, x: torch.Tensor, *, pkv: Any, cache: bool) -> torch.Tensor:
        layer = layers[layer_idx]
        layer_mask = _mask_for_layer(layer, masks)
        return _call_decoder_layer(
            layer,
            x,
            attention_mask=layer_mask,
            position_ids=position_ids,
            past_key_values=pkv,
            use_cache=cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            output_router_logits=kwargs.get("output_router_logits", False),
            extra_kwargs=kwargs,
        )

    def call_no_cache(layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        return call_layer(layer_idx, x, pkv=None, cache=False)

    def call_with_cache(layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        return call_layer(layer_idx, x, pkv=past_key_values, cache=use_cache)

    def run_block_prefill(x: torch.Tensor) -> torch.Tensor:
        for li in range(loop_start, loop_end):
            x = call_no_cache(li, x)
        return x

    def run_block_decode(x: torch.Tensor) -> torch.Tensor:
        saved = _snapshot_cache_lengths(past_key_values, loop_start, loop_end)
        for li in range(loop_start, loop_end):
            x = call_with_cache(li, x)
        _restore_cache_lengths(past_key_values, saved)
        return x

    def run_one_layer_prefill(li: int):
        return lambda z: call_no_cache(li, z)

    def run_one_layer_decode(li: int):
        def apply(z: torch.Tensor) -> torch.Tensor:
            saved = _snapshot_cache_lengths(past_key_values, li, li + 1)
            out = call_with_cache(li, z)
            _restore_cache_lengths(past_key_values, saved)
            return out

        return apply

    all_hidden_states = [] if output_hidden_states else None
    for idx in range(0, loop_start):
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        hidden_states = call_with_cache(idx, hidden_states)

    if not do_loop:
        for idx in range(loop_start, n_layers):
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states = call_with_cache(idx, hidden_states)
        return _finish_output(
            self,
            hidden_states,
            past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
            all_hidden_states=all_hidden_states,
        )

    run_block = run_block_decode if is_incremental else run_block_prefill
    run_one_layer = run_one_layer_decode if is_incremental else run_one_layer_prefill
    loop_entry_for_cache = hidden_states
    hidden_states = _apply_loop_window(
        hidden_states,
        run_block=run_block,
        run_one_layer=run_one_layer,
        loop_start=loop_start,
        loop_end=loop_end,
        strategy=self._tflt_strategy,
        loop_mode=self._tflt_loop_mode,
        K=self._tflt_K,
        anchor_beta=self._tflt_anchor_beta,
        ema_alpha=self._tflt_ema_alpha,
        momentum_beta=self._tflt_momentum_beta,
        anderson_m=self._tflt_anderson_m,
        anderson_beta=self._tflt_anderson_beta,
        per_step_alphas=self._tflt_per_step_alphas,
        halt_tau=self._tflt_halt_tau,
    )

    if use_cache and past_key_values is not None and self._tflt_cache_strategy != "none":
        cache_source = hidden_states if self._tflt_cache_strategy == "last" else loop_entry_for_cache
        for li in range(loop_start, loop_end):
            cache_source = call_with_cache(li, cache_source)

    for idx in range(loop_end, n_layers):
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        hidden_states = call_with_cache(idx, hidden_states)

    return _finish_output(
        self,
        hidden_states,
        past_key_values,
        use_cache=use_cache,
        return_dict=return_dict,
        all_hidden_states=all_hidden_states,
    )


def _apply_loop_window(
    hidden_states: torch.Tensor,
    *,
    run_block,
    run_one_layer,
    loop_start: int,
    loop_end: int,
    strategy: str,
    loop_mode: str,
    K: int,
    anchor_beta: float,
    ema_alpha: float,
    momentum_beta: float,
    anderson_m: int,
    anderson_beta: float,
    per_step_alphas: Sequence[float] | None,
    halt_tau: float | None,
) -> torch.Tensor:
    if strategy == "block_anchored":
        return anchored_euler(hidden_states, run_block, K=K, beta=anchor_beta)

    if strategy == "per_layer_anchored":
        x = hidden_states
        for layer_idx in range(loop_start, loop_end):
            x = anchored_euler(x, run_one_layer(layer_idx), K=K, beta=anchor_beta)
        return x

    if strategy == "uniform":
        if loop_mode != "block":
            raise ValueError("uniform strategy is only defined for block mode")
        running_sums = [None] * (loop_end - loop_start)
        counts = [0] * (loop_end - loop_start)
        x = hidden_states
        for _ in range(K):
            for offset, layer_idx in enumerate(range(loop_start, loop_end)):
                x = run_one_layer(layer_idx)(x)
                running_sums[offset] = x if running_sums[offset] is None else running_sums[offset] + x
                counts[offset] += 1
                x = running_sums[offset] / counts[offset]
        return x

    kwargs = {
        "strategy": strategy,
        "K": K,
        "ema_alpha": ema_alpha,
        "momentum_beta": momentum_beta,
        "anderson_m": anderson_m,
        "anderson_beta": anderson_beta,
        "per_step_alphas": per_step_alphas,
        "halt_tau": halt_tau,
        "anchor_beta": anchor_beta,
    }
    if loop_mode == "block":
        return iterate_map(hidden_states, run_block, **kwargs)
    if loop_mode == "layer":
        x = hidden_states
        for layer_idx in range(loop_start, loop_end):
            x = iterate_map(x, run_one_layer(layer_idx), **kwargs)
        return x
    raise ValueError(f"unknown loop_mode: {loop_mode!r}")


def _call_decoder_layer(
    layer: Any,
    hidden_states: torch.Tensor,
    *,
    attention_mask: Any,
    position_ids: torch.Tensor,
    past_key_values: Any,
    use_cache: bool,
    cache_position: torch.Tensor,
    position_embeddings: Any,
    output_attentions: bool | None,
    output_router_logits: bool | None,
    extra_kwargs: Mapping[str, Any],
) -> torch.Tensor:
    params, accepts_kwargs = _forward_signature(layer)
    call_kwargs: dict[str, Any] = {}

    def add(name: str, value: Any) -> None:
        if accepts_kwargs or name in params:
            call_kwargs[name] = value

    add("attention_mask", attention_mask)
    add("position_ids", position_ids)
    if "past_key_value" in params and "past_key_values" not in params:
        add("past_key_value", past_key_values)
    else:
        add("past_key_values", past_key_values)
    add("use_cache", use_cache)
    add("cache_position", cache_position)
    if position_embeddings is not None:
        add("position_embeddings", position_embeddings)
    if output_attentions is not None:
        add("output_attentions", output_attentions)
    if output_router_logits is not None:
        add("output_router_logits", output_router_logits)

    for key, value in extra_kwargs.items():
        if key not in call_kwargs and (accepts_kwargs or key in params):
            call_kwargs[key] = value

    out = layer(hidden_states, **call_kwargs)
    return _extract_hidden(out)


def _forward_signature(layer: Any) -> tuple[set[str], bool]:
    cached = getattr(layer, "_tflt_forward_signature", None)
    if cached is not None:
        return cached
    try:
        sig = inspect.signature(layer.forward)
    except (TypeError, ValueError):
        result = (set(), True)
    else:
        params = set(sig.parameters)
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
        )
        result = (params, accepts_kwargs)
    layer._tflt_forward_signature = result
    return result


def _extract_hidden(layer_output: Any) -> torch.Tensor:
    if isinstance(layer_output, torch.Tensor):
        return layer_output
    if isinstance(layer_output, tuple):
        return layer_output[0]
    if hasattr(layer_output, "last_hidden_state"):
        return layer_output.last_hidden_state
    raise TypeError(f"cannot extract hidden states from {type(layer_output)!r}")


def _build_causal_masks(
    model: Any,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: Any,
    cache_position: torch.Tensor,
    past_key_values: Any,
    position_ids: torch.Tensor,
) -> Any:
    if isinstance(attention_mask, dict):
        return attention_mask

    try:
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )
    except ImportError:
        create_causal_mask = None
        create_sliding_window_causal_mask = None

    if create_causal_mask is not None:
        mask_kwargs = {
            "config": model.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        try:
            masks = {"full_attention": create_causal_mask(**mask_kwargs)}
            if getattr(model, "has_sliding_layers", False) or getattr(model.config, "sliding_window", None):
                masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
            return masks
        except Exception:
            pass

    if hasattr(model, "_update_causal_mask"):
        for args in (
            (attention_mask, inputs_embeds, cache_position, past_key_values, False),
            (attention_mask, inputs_embeds, cache_position, past_key_values, None),
        ):
            try:
                mask = model._update_causal_mask(*args)
                return {"full_attention": mask, "sliding_attention": mask}
            except TypeError:
                continue

    return {"full_attention": attention_mask, "sliding_attention": attention_mask}


def _mask_for_layer(layer: Any, masks: Any) -> Any:
    if not isinstance(masks, dict):
        return masks
    attention_type = getattr(layer, "attention_type", "full_attention")
    return masks.get(attention_type, masks.get("full_attention"))


def _rotary_embeddings(model: Any, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> Any:
    if not hasattr(model, "rotary_emb"):
        return None
    try:
        return model.rotary_emb(hidden_states, position_ids)
    except TypeError:
        return None


def _finish_output(
    model: Any,
    hidden_states: torch.Tensor,
    past_key_values: Any,
    *,
    use_cache: bool,
    return_dict: bool,
    all_hidden_states: list[torch.Tensor] | None,
) -> Any:
    hidden_states = model.norm(hidden_states)
    if all_hidden_states is not None:
        all_hidden_states.append(hidden_states)
        all_hidden_states_tuple = tuple(all_hidden_states)
    else:
        all_hidden_states_tuple = None
    pkv = past_key_values if use_cache else None
    if not return_dict:
        values = (hidden_states, pkv, all_hidden_states_tuple)
        return tuple(value for value in values if value is not None)

    try:
        from transformers.modeling_outputs import BaseModelOutputWithPast, MoeModelOutputWithPast
    except ImportError:
        from transformers.modeling_outputs import BaseModelOutputWithPast

        MoeModelOutputWithPast = BaseModelOutputWithPast

    output_cls = MoeModelOutputWithPast if "moe" in getattr(model.config, "model_type", "") else BaseModelOutputWithPast
    return output_cls(
        last_hidden_state=hidden_states,
        past_key_values=pkv,
        hidden_states=all_hidden_states_tuple,
    )


def _new_dynamic_cache(config: Any) -> Any:
    from transformers.cache_utils import DynamicCache

    try:
        return DynamicCache(config=config)
    except TypeError:
        return DynamicCache()


def _cache_seq_length(cache: Any) -> int:
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        try:
            return int(cache.get_seq_length())
        except TypeError:
            return int(cache.get_seq_length(0))
    if hasattr(cache, "key_cache") and cache.key_cache:
        return int(cache.key_cache[0].shape[-2])
    return 0


def _snapshot_cache_lengths(cache: Any, start: int, end: int) -> list[tuple[int, int | None]]:
    return [(idx, _layer_cache_length(cache, idx)) for idx in range(start, end)]


def _restore_cache_lengths(cache: Any, saved: list[tuple[int, int | None]]) -> None:
    for idx, length in saved:
        if length is not None:
            _crop_layer_cache(cache, idx, length)


def _layer_cache_length(cache: Any, layer_idx: int) -> int | None:
    if cache is None:
        return None
    if hasattr(cache, "layers") and layer_idx < len(cache.layers):
        layer_cache = cache.layers[layer_idx]
        if hasattr(layer_cache, "get_seq_length"):
            return int(layer_cache.get_seq_length())
    if hasattr(cache, "get_seq_length"):
        try:
            return int(cache.get_seq_length(layer_idx))
        except TypeError:
            return int(cache.get_seq_length())
    if hasattr(cache, "key_cache") and len(cache.key_cache) > layer_idx:
        return int(cache.key_cache[layer_idx].shape[-2])
    return None


def _crop_layer_cache(cache: Any, layer_idx: int, length: int) -> None:
    if hasattr(cache, "layers") and layer_idx < len(cache.layers):
        layer_cache = cache.layers[layer_idx]
        if hasattr(layer_cache, "crop"):
            layer_cache.crop(length)
            return
    if hasattr(cache, "crop"):
        cache.crop(length)
        return
    raise RuntimeError(
        "decode loop needs a DynamicCache implementation with crop support. "
        "Use decode_mode='bypass' for loglikelihood evaluation."
    )


def _should_loop(
    *,
    is_incremental: bool,
    decode_mode: str,
    decode_count: int,
    decode_first_n: int,
) -> bool:
    if not is_incremental:
        return True
    if decode_mode == "bypass":
        return False
    if decode_mode == "full":
        return True
    if decode_mode == "first_n":
        return decode_count <= decode_first_n
    raise ValueError(f"unknown decode_mode: {decode_mode!r}")


def _loop_bounds(indices: Sequence[int]) -> tuple[int, int]:
    sorted_indices = sorted(int(idx) for idx in indices)
    start = sorted_indices[0]
    end = sorted_indices[-1] + 1
    if sorted_indices != list(range(start, end)):
        raise ValueError(f"loop indices must be contiguous, got {indices}")
    return start, end


def _validate_config(config: LoopConfig) -> None:
    _loop_bounds(config.loop_indices)
    if config.K <= 0:
        raise ValueError("K must be positive")
    if config.loop_mode not in {"block", "layer"}:
        raise ValueError("loop_mode must be block or layer")
    if config.cache_strategy not in {"first", "last", "none"}:
        raise ValueError("cache_strategy must be first, last, or none")
    if config.decode_mode not in {"bypass", "full", "first_n"}:
        raise ValueError("decode_mode must be bypass, full, or first_n")

