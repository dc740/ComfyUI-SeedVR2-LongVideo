"""Route SeedVR2 DiT attention through the gfx906 tiled HIP kernel.

Installs a ``container_function`` on ``optimized_attention`` — the seam
``comfy/ldm/modules/attention.py`` reserves for containerized q/k/v. Eligible
calls (gfx906, fp16/bf16 on CUDA, head_dim<=64 mult-of-8 or ==128, no mask/GQA)
run the tiled kernel, which never materializes the SxS score matrix that the
MATH SDPA backend allocates (dozens of GB for 4K SeedVR2 window runs).
Everything else falls through to the stock callable, so unsupported shapes and
other models keep their original path. The kernel .so loads lazily on the
first eligible call; off gfx906 nothing is installed.
"""

import logging
import os

import torch

from . import gfx906_attn


logger = logging.getLogger("SeedVR2-LongVideo")
_LOG_SHAPES = os.environ.get("SEEDVR2_GFX906_LOG_SHAPES") == "1"
_seen_shapes = set()

_stock_attention = None  # wrapped optimized_attention, used for fallback


def _eligible(qt, kt, vt, heads, mask, kwargs):
    if mask is not None or kwargs.get("enable_gqa"):
        return False
    if qt.dtype not in (torch.float16, torch.bfloat16) or not qt.is_cuda:
        return False
    if qt.ndim != 4 or kt.shape != vt.shape:
        return False
    if qt.shape[0] != kt.shape[0] or qt.shape[1] != kt.shape[1] or \
            qt.shape[3] != kt.shape[3] or qt.shape[1] != heads:
        return False
    d = qt.shape[3]
    return d == 128 or (d <= 64 and d % 8 == 0)


def _container_attn(q, k, v, heads, mask=None, attn_precision=None,
                    skip_reshape=False, skip_output_reshape=False, **kwargs):
    qt, kt, vt = q.peek(), k.peek(), v.peek()
    mod = None
    if skip_reshape and _eligible(qt, kt, vt, heads, mask, kwargs):
        mod = gfx906_attn.load_attention()
    if mod is None:
        return _stock_attention(
            q.take(), k.take(), v.take(), heads, mask=mask,
            attn_precision=attn_precision, skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape, **kwargs)
    if _LOG_SHAPES and tuple(qt.shape) not in _seen_shapes:
        _seen_shapes.add(tuple(qt.shape))
        logger.info("gfx906 attention: %s %s", tuple(qt.shape), qt.dtype)
    qt, kt, vt = q.take(), k.take(), v.take()
    scale = kwargs.get("scale")
    if scale is None:
        scale = qt.shape[-1] ** -0.5  # sdpa default
    out_dtype = qt.dtype
    if out_dtype != torch.float16:
        qt, kt, vt = qt.half(), kt.half(), vt.half()
    out = mod.attn(qt, kt, vt, scale)
    if out_dtype != torch.float16:
        out = out.to(out_dtype)
    if skip_output_reshape:
        return out
    b = out.shape[0]
    return out.transpose(1, 2).reshape(b, -1, heads * out.shape[-1])


def install():
    """Hook ``optimized_attention.container_function`` once. No-op off gfx906."""
    if not gfx906_attn.is_gfx906():
        return
    global _stock_attention
    try:
        from comfy.ldm.modules import attention as _attn
    except Exception as exc:  # noqa: BLE001 - comfy not on sys.path (e.g. tests)
        logger.warning("gfx906 attention hook skipped: %s", exc)
        return
    opt = _attn.optimized_attention
    if _stock_attention is opt:
        return
    if not hasattr(opt, "container_function"):
        return
    if opt.container_function is not None:
        logger.warning("optimized_attention.container_function already set; "
                       "gfx906 attention hook not installed")
        return
    _stock_attention = opt
    opt.container_function = _container_attn
    logger.info("gfx906 tiled attention installed (SeedVR2 DiT)")
