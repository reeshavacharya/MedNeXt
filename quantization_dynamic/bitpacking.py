"""Bit packing utilities for low-bit integer storage.

This module provides helpers to *store* logical INT6 and INT4 values
inside ``torch.int8`` tensors. GPUs generally do not support native
INT6, and INT4 support is limited, so both formats are encoded into
8-bit containers.

Supported logical formats:

- INT6: range [-32, 31], one value per byte, lower 6 bits used
- INT4: range [-8, 7], two values per byte (two 4-bit nibbles)
"""

from __future__ import annotations

from typing import Optional

import logging

import numpy as np
import torch


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# INT6 packing utilities
# ---------------------------------------------------------------------------


def pack_int6(tensor: torch.Tensor) -> torch.Tensor:
    """Pack logical INT6 values into an int8 container tensor.

    Parameters
    ----------
    tensor:
        Tensor containing signed values in (or to be clamped to)
        the range [-32, 31]. Can be any integer or floating dtype.

    Returns
    -------
    packed:
        ``torch.int8`` tensor with the same shape as ``tensor`` where
        each element stores the lower 6 bits of the corresponding value.
    """

    logger.info("Packing INT6 tensor into int8 container")
    # Convert to integer and clamp to valid INT6 range
    q = tensor.to(torch.int16)
    q = torch.clamp(q, -32, 31)
    # Mask lower 6 bits and store as int8
    packed = (q & 0x3F).to(torch.int8)
    return packed


def unpack_int6(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int8 container tensor back to logical INT6 values.

    Parameters
    ----------
    packed:
        ``torch.int8`` tensor produced by :func:`pack_int6`.

    Returns
    -------
    values:
        ``torch.int8`` tensor with logical INT6 values in [-32, 31].
    """

    logger.info("Unpacking INT6 values from int8 container")
    val = packed.to(torch.int16) & 0x3F
    neg_mask = val >= 32
    val[neg_mask] -= 64
    return val.to(torch.int8)


# ---------------------------------------------------------------------------
# INT4 packing utilities
# ---------------------------------------------------------------------------


def pack_int4(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Pack logical INT4 values into an int8 container along a dimension.

    Two INT4 values (range [-8, 7]) are packed into one byte using:

        packed = (v1 & 0xF) | ((v2 & 0xF) << 4)

    The packing is performed along the specified dimension ``dim``.
    If the length along that dimension is odd, a zero-valued element
    is implicitly padded at the end for packing purposes.

    Parameters
    ----------
    tensor:
        Tensor containing signed values in (or to be clamped to)
        the range [-8, 7].
    dim:
        Dimension along which to pack pairs of values. Default: -1

    Returns
    -------
    packed:
        ``torch.int8`` tensor with the same shape as ``tensor`` except
        that the size of the packed dimension is ``ceil(N / 2)``.
    """

    logger.info("Packing INT4 tensor into int8 container along dim=%d", dim)

    if tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        q = torch.round(tensor).to(torch.int16)
    else:
        q = tensor.to(torch.int16)

    q = torch.clamp(q, -8, 7)
    # Move packing dimension to the last axis for easier handling
    dim = dim if dim >= 0 else tensor.dim() + dim
    perm = list(range(tensor.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    q = q.permute(*perm).contiguous()

    orig_shape = q.shape
    length = orig_shape[-1]
    if length % 2 != 0:
        pad = torch.zeros(orig_shape[:-1] + (1,), dtype=q.dtype, device=q.device)
        q = torch.cat([q, pad], dim=-1)
        length += 1

    q = q.view(-1, length)
    low = q[:, 0::2] & 0xF
    high = q[:, 1::2] & 0xF
    packed = (low | (high << 4)).to(torch.int8)

    packed = packed.view(*orig_shape[:-1], length // 2)
    # Inverse permute to restore original dimension order (with halved dim)
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    packed = packed.permute(*inv_perm).contiguous()
    return packed


def unpack_int4(packed: torch.Tensor, dim: int = -1, *, orig_length: Optional[int] = None) -> torch.Tensor:
    """Unpack int8 container back to logical INT4 values along a dimension.

    Parameters
    ----------
    packed:
        ``torch.int8`` tensor produced by :func:`pack_int4`.
    dim:
        Dimension along which the original packing was performed.
    orig_length:
        Optional original length along ``dim`` before packing. If not
        provided, the unpacked length is assumed to be ``2 * size(dim)``.

    Returns
    -------
    values:
        ``torch.int8`` tensor with unpacked INT4 values in [-8, 7].
    """

    logger.info("Unpacking INT4 values from int8 container along dim=%d", dim)

    # Move target dimension to last axis
    dim = dim if dim >= 0 else packed.dim() + dim
    perm = list(range(packed.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    q = packed.permute(*perm).contiguous()

    orig_shape = q.shape
    length = orig_shape[-1]
    q = q.view(-1, length).to(torch.int16)

    low = q & 0xF
    high = (q >> 4) & 0xF

    # Convert from unsigned [0, 15] to signed [-8, 7]
    low[low >= 8] -= 16
    high[high >= 8] -= 16

    vals = torch.empty(q.size(0), length * 2, dtype=torch.int8, device=q.device)
    vals[:, 0::2] = low.to(torch.int8)
    vals[:, 1::2] = high.to(torch.int8)

    unpacked = vals.view(*orig_shape[:-1], length * 2)
    if orig_length is not None and orig_length < length * 2:
        unpacked = unpacked[..., :orig_length]

    # Inverse permute to restore original dimension order
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    unpacked = unpacked.permute(*inv_perm).contiguous()
    return unpacked
