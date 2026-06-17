"""Observation schema helpers for rehabilitation environments."""

from __future__ import annotations

import numpy as np
import torch as th

OBS_SCALAR_DIM = 12
DEFAULT_HISTORY_LENGTH = 16
HISTORY_CHANNELS = 2
INTERACTION_HISTORY_CHANNELS = 8


def history_dim(history_length: int, history_channels: int = HISTORY_CHANNELS) -> int:
    return int(history_length) * int(history_channels)


def obs_dim(
    history_length: int = DEFAULT_HISTORY_LENGTH,
    opponent_id_dim: int = 0,
    history_channels: int = HISTORY_CHANNELS,
) -> int:
    return OBS_SCALAR_DIM + history_dim(history_length, history_channels) + int(opponent_id_dim)


def history_slice(
    history_length: int = DEFAULT_HISTORY_LENGTH,
    history_channels: int = HISTORY_CHANNELS,
) -> slice:
    return slice(OBS_SCALAR_DIM, OBS_SCALAR_DIM + history_dim(history_length, history_channels))


def id_slice(
    history_length: int = DEFAULT_HISTORY_LENGTH,
    history_channels: int = HISTORY_CHANNELS,
) -> slice:
    start = OBS_SCALAR_DIM + history_dim(history_length, history_channels)
    return slice(start, None)


def infer_history_length(
    total_obs_dim: int,
    opponent_id_dim: int = 0,
    history_channels: int = HISTORY_CHANNELS,
) -> int:
    history_size = int(total_obs_dim) - OBS_SCALAR_DIM - int(opponent_id_dim)
    if history_size < 0 or history_size % int(history_channels) != 0:
        raise ValueError(
            f"Cannot infer history length from obs_dim={total_obs_dim}, "
            f"opponent_id_dim={opponent_id_dim}, history_channels={history_channels}"
        )
    return history_size // int(history_channels)


def split_flat_observation(
    observations,
    history_length: int = DEFAULT_HISTORY_LENGTH,
    include_id_as_scalar: bool = True,
    history_channels: int = HISTORY_CHANNELS,
):
    scalar_part = observations[:, :OBS_SCALAR_DIM]
    history_part = observations[:, history_slice(history_length, history_channels)]
    id_part = observations[:, id_slice(history_length, history_channels)]
    if include_id_as_scalar and id_part.shape[1] > 0:
        scalar_part = th.cat([scalar_part, id_part], dim=1)
    return scalar_part, history_part, id_part


def adapt_history_obs(obs, target_obs_dim: int, opponent_id_dim: int = 0):
    arr = np.asarray(obs, dtype=np.float32)
    if arr.shape[-1] == target_obs_dim:
        return arr.astype(np.float32, copy=False)

    target_history_length = infer_history_length(target_obs_dim, opponent_id_dim)
    target_history_size = history_dim(target_history_length)

    scalar = arr[..., :OBS_SCALAR_DIM]
    source_history = arr[..., OBS_SCALAR_DIM:]
    if opponent_id_dim > 0:
        source_history = source_history[..., :-opponent_id_dim]

    source_history_size = source_history.shape[-1]
    if source_history_size >= target_history_size:
        target_history = source_history[..., -target_history_size:]
    else:
        pad_shape = source_history.shape[:-1] + (target_history_size - source_history_size,)
        pad = np.zeros(pad_shape, dtype=np.float32)
        target_history = np.concatenate([pad, source_history], axis=-1)

    parts = [scalar, target_history]
    if opponent_id_dim > 0:
        parts.append(np.zeros(arr.shape[:-1] + (opponent_id_dim,), dtype=np.float32))
    adapted = np.concatenate(parts, axis=-1).astype(np.float32)
    if adapted.shape[-1] != target_obs_dim:
        raise ValueError(f"Adapted obs has shape {adapted.shape[-1]}, expected {target_obs_dim}")
    return adapted


def model_obs_dim(model) -> int | None:
    space = getattr(model, "observation_space", None)
    shape = getattr(space, "shape", None)
    if shape is None and hasattr(model, "get_env"):
        env = model.get_env()
        if env is not None:
            shape = getattr(getattr(env, "observation_space", None), "shape", None)
    if shape is None:
        return None
    return int(shape[0])
