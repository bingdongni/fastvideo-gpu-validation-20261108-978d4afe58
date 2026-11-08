"""Load the pinned FastVideo Python backend with GPU SDPA in place of FA kernels.

This deliberately validates the real backend's adapter logic, not an actual
FlashAttention binary. It is only for an environment without CUDA build tools.
"""
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(os.environ['FASTVIDEO_SOURCE_ROOT'])


def module(name, **attributes):
    result = types.ModuleType(name)
    result.__dict__.update(attributes)
    sys.modules[name] = result
    parent, _, child = name.rpartition('.')
    if parent in sys.modules:
        setattr(sys.modules[parent], child, result)
    return result


def package(name, path):
    return module(name, __path__=[str(path)])


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning_once(self, *args, **kwargs):
        pass


class _Flag:
    def get(self):
        return None


def _attention(q, k, v, query_padding_mask, key_padding_mask,
               causal=False, dropout_p=0.0, softmax_scale=None, **kwargs):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    if os.environ.get('FASTVIDEO_TEST_FA_VERSION') == '3' and q.shape[1] != k.shape[1]:
        raise TypeError('simulated FA3 varlen function does not accept dropout_p')
    result = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=key_padding_mask[:, None, None, :], dropout_p=dropout_p,
        is_causal=causal, scale=softmax_scale,
    ).transpose(1, 2)
    return result * query_padding_mask[:, :, None, None]


def _self_attention(qkv, mask, **kwargs):
    q, k, v = qkv.unbind(dim=2)
    return _attention(q, k, v, mask, mask, **kwargs)


def pytest_configure(config):
    package('fastvideo', ROOT / 'fastvideo')
    package('fastvideo.attention', ROOT / 'fastvideo' / 'attention')
    package('fastvideo.attention.backends', ROOT / 'fastvideo' / 'attention' / 'backends')
    package('fastvideo.attention.utils', ROOT / 'fastvideo' / 'attention' / 'utils')
    module('fastvideo.envs', FASTVIDEO_NVFP4_FA4_QK_TRANSFORM=_Flag())
    module('fastvideo.logger', init_logger=lambda name: _Logger())
    module('fastvideo.attention.utils.cudnn_sdpa', cudnn_sdpa_bshd=None,
           cudnn_sdpa_supported=lambda *args: False)
    module('fastvideo.attention.utils.flash_attn_default', fa_version='2',
           flash_attn_func_compilable=None)
    module('fastvideo.attention.backends.attn_qat_infer', _resolve_fa4_pv_mode=lambda _: None)
    module('fastvideo.attention.utils.nvfp4_qk_precondition',
           parse_qk_transform=lambda _: (False, False), precondition_qk=None)
    module('fastvideo.attention.utils.nvfp4_fused_qk_quant',
           fused_qk_quant_available=lambda: False, fused_qk_quant_supported=lambda *args: False,
           nvfp4_fused_qk_quantize_fa4=None)
    module('fastvideo.attention.utils.flash_attn_no_pad',
           _FA_VARLEN_VERSION=os.environ.get('FASTVIDEO_TEST_FA_VERSION', '2'),
           flash_attn_no_pad_compilable=_self_attention,
           flash_attn_varlen_qk_no_pad_compilable=_attention)
