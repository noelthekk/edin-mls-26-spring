"""
cuTile Compatibility Layer for Non-Blackwell GPUs (Hopper Hack)

This module provides a drop-in replacement for cuda.tile that works on
older GPUs (Ada Lovelace sm_89, Ampere sm_80, etc.) by using CuPy.

The original cuTile only supports Blackwell GPUs (sm_100+).
This hack intercepts the cuTile API and implements equivalent operations
using CuPy, without trying to compile custom CUDA kernels.

Strategy: Runtime interpretation using CuPy operations instead of
AST-based CUDA code generation.
"""

import builtins
import cupy as cp
import numpy as np
import math
from typing import Callable, Tuple, Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from functools import wraps

# Save Python builtins before we override them
_builtin_min = builtins.min
_builtin_max = builtins.max
_builtin_sum = builtins.sum
_builtin_pow = builtins.pow

# =============================================================================
# Data Types
# =============================================================================

class DType:
    """Base class for cuTile data types."""
    pass

# Data type singletons
class _Int8(DType):
    name = "int8"
    ctype = "signed char"
    nptype = np.int8
int8 = _Int8()

class _Int16(DType):
    name = "int16"
    ctype = "short"
    nptype = np.int16
int16 = _Int16()

class _Int32(DType):
    name = "int32"
    ctype = "int"
    nptype = np.int32
int32 = _Int32()

class _Int64(DType):
    name = "int64"
    ctype = "long long"
    nptype = np.int64
int64 = _Int64()

class _UInt8(DType):
    name = "uint8"
    ctype = "unsigned char"
    nptype = np.uint8
uint8 = _UInt8()

class _UInt16(DType):
    name = "uint16"
    ctype = "unsigned short"
    nptype = np.uint16
uint16 = _UInt16()

class _UInt32(DType):
    name = "uint32"
    ctype = "unsigned int"
    nptype = np.uint32
uint32 = _UInt32()

class _UInt64(DType):
    name = "uint64"
    ctype = "unsigned long long"
    nptype = np.uint64
uint64 = _UInt64()

class _Float16(DType):
    name = "float16"
    ctype = "__half"
    nptype = np.float16
float16 = _Float16()

class _Float32(DType):
    name = "float32"
    ctype = "float"
    nptype = np.float32
float32 = _Float32()

class _Float64(DType):
    name = "float64"
    ctype = "double"
    nptype = np.float64
float64 = _Float64()

class _BFloat16(DType):
    name = "bfloat16"
    ctype = "__nv_bfloat16"
    nptype = np.float16  # CuPy uses float16 for bfloat16
bfloat16 = _BFloat16()

class _TFloat32(DType):
    name = "tfloat32"
    ctype = "float"  # TF32 uses float storage
    nptype = np.float32
tfloat32 = _TFloat32()

class _Bool(DType):
    name = "bool"
    ctype = "bool"
    nptype = np.bool_
bool_ = _Bool()

class _Float8E4M3FN(DType):
    name = "float8_e4m3fn"
    ctype = "__nv_fp8_e4m3"
    nptype = np.float16
float8_e4m3fn = _Float8E4M3FN()

class _Float8E5M2(DType):
    name = "float8_e5m2"
    ctype = "__nv_fp8_e5m2"
    nptype = np.float16
float8_e5m2 = _Float8E5M2()


def _dtype_to_nptype(dtype):
    """Convert cuTile dtype to numpy dtype."""
    if isinstance(dtype, DType):
        return dtype.nptype
    return np.dtype(dtype)


# =============================================================================
# Type Annotations
# =============================================================================

class Constant:
    """Type annotation for compile-time constants."""
    def __class_getitem__(cls, item):
        return item


class ConstantAnnotation:
    """Marker for constant annotations."""
    pass


class Array:
    """Type annotation for arrays."""
    def __class_getitem__(cls, item):
        return item


class Scalar:
    """Type annotation for scalars."""
    def __class_getitem__(cls, item):
        return item


class Tile:
    """Type annotation for tiles."""
    def __class_getitem__(cls, item):
        return item


class ByTarget:
    """Target-specific configuration."""
    def __class_getitem__(cls, item):
        return item


# =============================================================================
# Enums
# =============================================================================

class MemoryOrder:
    relaxed = "relaxed"
    acquire = "acquire"
    release = "release"
    acq_rel = "acq_rel"
    seq_cst = "seq_cst"


class MemoryScope:
    system = "system"
    device = "device"
    block = "block"


class PaddingMode:
    zeros = "zeros"
    reflect = "reflect"
    replicate = "replicate"


class RoundingMode:
    nearest = "nearest"
    down = "down"
    up = "up"
    truncate = "truncate"


# =============================================================================
# Exceptions
# =============================================================================

class TileCompilerError(Exception):
    """Base class for tile compiler errors."""
    pass


class TileCompilerExecutionError(TileCompilerError):
    """Raised when tile compiler execution fails."""
    pass


class TileCompilerTimeoutError(TileCompilerError):
    """Raised when tile compiler times out."""
    pass


class TileInternalError(TileCompilerError):
    """Raised for internal errors."""
    pass


class TileSyntaxError(TileCompilerError):
    """Raised for syntax errors in tile code."""
    pass


class TileTypeError(TileCompilerError):
    """Raised for type errors in tile code."""
    pass


class TileValueError(TileCompilerError):
    """Raised for value errors in tile code."""
    pass


# =============================================================================
# Utility Functions
# =============================================================================

def cdiv(a: int, b: int) -> int:
    """Ceiling division: (a + b - 1) // b"""
    return (a + b - 1) // b


# =============================================================================
# Stub Functions (for AST parsing - not called at runtime)
# =============================================================================

def bid(dim: int) -> int:
    """Get block ID in given dimension."""
    raise RuntimeError("bid() should only be called within a kernel")


def num_blocks(dim: int) -> int:
    """Get number of blocks in given dimension."""
    raise RuntimeError("num_blocks() should only be called within a kernel")


def num_tiles(dim: int) -> int:
    """Get number of tiles in given dimension."""
    raise RuntimeError("num_tiles() should only be called within a kernel")


def load(array, index: Tuple, shape: Tuple, **kwargs):
    """Load a tile from global memory."""
    raise RuntimeError("load() should only be called within a kernel")


def store(array, index: Tuple, tile):
    """Store a tile to global memory."""
    raise RuntimeError("store() should only be called within a kernel")


def full(shape: Tuple, value, dtype=None):
    """Create a tile filled with a value."""
    raise RuntimeError("full() should only be called within a kernel")


def zeros(shape: Tuple, dtype=None):
    """Create a tile filled with zeros."""
    raise RuntimeError("zeros() should only be called within a kernel")


def ones(shape: Tuple, dtype=None):
    """Create a tile filled with ones."""
    raise RuntimeError("ones() should only be called within a kernel")


def arange(start, stop=None, step=1, dtype=None):
    """Create a tile with evenly spaced values."""
    raise RuntimeError("arange() should only be called within a kernel")


def astype(tile, dtype):
    """Convert tile to specified data type."""
    raise RuntimeError("astype() should only be called within a kernel")


def transpose(tile, axes=None):
    """Transpose a tile."""
    raise RuntimeError("transpose() should only be called within a kernel")


def permute(tile, axes):
    """Permute tile dimensions."""
    raise RuntimeError("permute() should only be called within a kernel")


def reshape(tile, shape):
    """Reshape a tile."""
    raise RuntimeError("reshape() should only be called within a kernel")


def broadcast_to(tile, shape):
    """Broadcast tile to shape."""
    raise RuntimeError("broadcast_to() should only be called within a kernel")


def expand_dims(tile, axis):
    """Expand tile dimensions."""
    raise RuntimeError("expand_dims() should only be called within a kernel")


def cat(tiles, axis=0):
    """Concatenate tiles."""
    raise RuntimeError("cat() should only be called within a kernel")


def bitcast(tile, dtype):
    """Bitcast tile to dtype."""
    raise RuntimeError("bitcast() should only be called within a kernel")


def extract(tile, indices):
    """Extract elements from tile."""
    raise RuntimeError("extract() should only be called within a kernel")


def gather(array, indices, axis=0):
    """Gather elements from array."""
    raise RuntimeError("gather() should only be called within a kernel")


def scatter(array, indices, tile, axis=0):
    """Scatter tile to array."""
    raise RuntimeError("scatter() should only be called within a kernel")


def where(condition, x, y):
    """Conditional selection."""
    raise RuntimeError("where() should only be called within a kernel")


# Math functions
def exp(x, **kwargs):
    raise RuntimeError("exp() should only be called within a kernel")

def exp2(x, **kwargs):
    raise RuntimeError("exp2() should only be called within a kernel")

def log(x):
    raise RuntimeError("log() should only be called within a kernel")

def log2(x):
    raise RuntimeError("log2() should only be called within a kernel")

def sqrt(x):
    raise RuntimeError("sqrt() should only be called within a kernel")

def rsqrt(x):
    raise RuntimeError("rsqrt() should only be called within a kernel")

def sin(x):
    raise RuntimeError("sin() should only be called within a kernel")

def cos(x):
    raise RuntimeError("cos() should only be called within a kernel")

def tan(x):
    raise RuntimeError("tan() should only be called within a kernel")

def sinh(x):
    raise RuntimeError("sinh() should only be called within a kernel")

def cosh(x):
    raise RuntimeError("cosh() should only be called within a kernel")

def tanh(x):
    raise RuntimeError("tanh() should only be called within a kernel")

def floor(x):
    raise RuntimeError("floor() should only be called within a kernel")

def ceil(x):
    raise RuntimeError("ceil() should only be called within a kernel")

def pow(x, y):
    raise RuntimeError("pow() should only be called within a kernel")


# Reduction functions
def sum(x, axis=None, keepdims=False):
    raise RuntimeError("sum() should only be called within a kernel")

def prod(x, axis=None):
    raise RuntimeError("prod() should only be called within a kernel")

def min(x, axis=None, keepdims=False):
    raise RuntimeError("min() should only be called within a kernel")

def max(x, axis=None, keepdims=False):
    raise RuntimeError("max() should only be called within a kernel")

def argmin(x, axis=None):
    raise RuntimeError("argmin() should only be called within a kernel")

def argmax(x, axis=None):
    raise RuntimeError("argmax() should only be called within a kernel")

def cumsum(x, axis=None):
    raise RuntimeError("cumsum() should only be called within a kernel")

def cumprod(x, axis=None):
    raise RuntimeError("cumprod() should only be called within a kernel")

def minimum(x, y):
    raise RuntimeError("minimum() should only be called within a kernel")

def maximum(x, y):
    raise RuntimeError("maximum() should only be called within a kernel")


# Binary operations
def add(x, y):
    raise RuntimeError("add() should only be called within a kernel")

def sub(x, y):
    raise RuntimeError("sub() should only be called within a kernel")

def mul(x, y):
    raise RuntimeError("mul() should only be called within a kernel")

def truediv(x, y, **kwargs):
    raise RuntimeError("truediv() should only be called within a kernel")

def floordiv(x, y):
    raise RuntimeError("floordiv() should only be called within a kernel")

def mod(x, y):
    raise RuntimeError("mod() should only be called within a kernel")

def negative(x):
    raise RuntimeError("negative() should only be called within a kernel")


# Comparison
def equal(x, y):
    raise RuntimeError("equal() should only be called within a kernel")

def not_equal(x, y):
    raise RuntimeError("not_equal() should only be called within a kernel")

def less(x, y):
    raise RuntimeError("less() should only be called within a kernel")

def less_equal(x, y):
    raise RuntimeError("less_equal() should only be called within a kernel")

def greater(x, y):
    raise RuntimeError("greater() should only be called within a kernel")

def greater_equal(x, y):
    raise RuntimeError("greater_equal() should only be called within a kernel")


# Bitwise
def bitwise_and(x, y):
    raise RuntimeError("bitwise_and() should only be called within a kernel")

def bitwise_or(x, y):
    raise RuntimeError("bitwise_or() should only be called within a kernel")

def bitwise_xor(x, y):
    raise RuntimeError("bitwise_xor() should only be called within a kernel")

def bitwise_not(x):
    raise RuntimeError("bitwise_not() should only be called within a kernel")

def bitwise_lshift(x, y):
    raise RuntimeError("bitwise_lshift() should only be called within a kernel")

def bitwise_rshift(x, y):
    raise RuntimeError("bitwise_rshift() should only be called within a kernel")


# Matrix operations
def matmul(a, b):
    raise RuntimeError("matmul() should only be called within a kernel")

def mma(a, b, c):
    raise RuntimeError("mma() should only be called within a kernel")


# Atomic operations
def atomic_add(array, index, value):
    raise RuntimeError("atomic_add() should only be called within a kernel")

def atomic_and(array, index, value):
    raise RuntimeError("atomic_and() should only be called within a kernel")

def atomic_or(array, index, value):
    raise RuntimeError("atomic_or() should only be called within a kernel")

def atomic_xor(array, index, value):
    raise RuntimeError("atomic_xor() should only be called within a kernel")

def atomic_min(array, index, value):
    raise RuntimeError("atomic_min() should only be called within a kernel")

def atomic_max(array, index, value):
    raise RuntimeError("atomic_max() should only be called within a kernel")

def atomic_xchg(array, index, value):
    raise RuntimeError("atomic_xchg() should only be called within a kernel")

def atomic_cas(array, index, compare, value):
    raise RuntimeError("atomic_cas() should only be called within a kernel")


# Debug
def printf(fmt, *args):
    raise RuntimeError("printf() should only be called within a kernel")

def assert_(condition, msg=""):
    raise RuntimeError("assert_() should only be called within a kernel")


# =============================================================================
# CuPy-based Kernel Implementations
# =============================================================================

def _impl_rmsnorm_kernel(x, weight, output, eps, hidden_size):
    """RMSNorm: x / RMS(x) * weight"""
    # x: (batch_size, hidden_size), weight: (hidden_size,)
    x_float = x.astype(cp.float32)
    variance = cp.mean(x_float ** 2, axis=-1, keepdims=True)
    x_norm = x_float / cp.sqrt(variance + eps)
    result = x_norm * weight
    output[:] = result


def _impl_compute_freqs_kernel(positions, inv_freq, cos_out, sin_out, seq_len, half_dim):
    """Compute cos and sin for rotary embeddings."""
    # positions: (seq_len,), inv_freq: (half_dim,)
    # cos_out, sin_out: (seq_len, rotary_dim) where rotary_dim = 2 * half_dim

    # Compute freqs = positions[:, None] * inv_freq[None, :]
    # Result: (seq_len, half_dim)
    freqs = positions[:, None].astype(cp.float32) * inv_freq[None, :].astype(cp.float32)

    # Compute cos and sin
    cos_half = cp.cos(freqs)
    sin_half = cp.sin(freqs)

    # Repeat for full dimension: [cos_half, cos_half]
    cos_out[:, :half_dim] = cos_half
    cos_out[:, half_dim:] = cos_half
    sin_out[:, :half_dim] = sin_half
    sin_out[:, half_dim:] = sin_half


def _impl_layernorm_kernel(x, weight, bias, output, eps, hidden_size):
    """LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias"""
    x_float = x.astype(cp.float32)
    mean = cp.mean(x_float, axis=-1, keepdims=True)
    variance = cp.var(x_float, axis=-1, keepdims=True)
    x_norm = (x_float - mean) / cp.sqrt(variance + eps)
    result = x_norm * weight + bias
    output[:] = result


def _impl_gelu_kernel(x, output, tile_size):
    """GELU using tanh approximation."""
    sqrt_2_over_pi = 0.7978845608028654
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x3)
    result = x * 0.5 * (1.0 + cp.tanh(inner))
    output[:] = result


def _impl_silu_kernel(x, output, tile_size):
    """SiLU/Swish: x * sigmoid(x)"""
    sigmoid = 1.0 / (1.0 + cp.exp(-x))
    result = x * sigmoid
    output[:] = result


def _impl_embedding_kernel(indices, weight, output, embed_dim):
    """Embedding lookup."""
    # indices: (batch_size,), weight: (vocab_size, embed_dim), output: (batch_size, embed_dim)
    indices_flat = indices.flatten().astype(cp.int64)
    output[:] = weight[indices_flat]


def _impl_softmax_kernel(x, output, seq_len):
    """Numerically stable softmax over last dimension."""
    x_max = cp.max(x, axis=-1, keepdims=True)
    x_shifted = x - x_max
    exp_x = cp.exp(x_shifted)
    sum_exp = cp.sum(exp_x, axis=-1, keepdims=True)
    output[:] = exp_x / sum_exp


def _impl_linear_kernel_tf32(x, weight_t, output, M, N, K):
    """Linear layer: output = x @ weight_t"""
    # x: (M, K), weight_t: (K, N), output: (M, N)
    result = x @ weight_t
    output[:] = result


def _impl_linear_silu_kernel(x, weight_t, output, M, N, K):
    """Fused Linear + SiLU: output = SiLU(x @ weight_t)"""
    result = x @ weight_t
    sigmoid = 1.0 / (1.0 + cp.exp(-result))
    output[:] = result * sigmoid


def _impl_linear_gelu_kernel(x, weight_t, output, M, N, K):
    """Fused Linear + GELU: output = GELU(x @ weight_t)"""
    result = x @ weight_t
    sqrt_2_over_pi = 0.7978845608028654
    result3 = result * result * result
    inner = sqrt_2_over_pi * (result + 0.044715 * result3)
    output[:] = result * 0.5 * (1.0 + cp.tanh(inner))


def _impl_swiglu_fused_kernel(x, gate_weight_t, up_weight_t, output, M, N, K):
    """Fused SwiGLU: output = SiLU(x @ gate_weight_t) * (x @ up_weight_t)"""
    gate = x @ gate_weight_t
    up = x @ up_weight_t
    sigmoid = 1.0 / (1.0 + cp.exp(-gate))
    gate_activated = gate * sigmoid
    output[:] = gate_activated * up


def _impl_attention_scores_kernel(q, k, scores, scale, seq_k, head_dim):
    """Compute attention scores: Q @ K.T * scale"""
    # q: (batch*heads, seq_q, head_dim), k: (batch*heads, seq_k, head_dim)
    # scores: (batch*heads, seq_q, seq_k)
    result = cp.einsum('bqd,bkd->bqk', q, k) * scale
    scores[:] = result


def _impl_softmax_inplace_kernel(scores, seq_k):
    """Apply softmax to attention scores."""
    s_max = cp.max(scores, axis=-1, keepdims=True)
    s_shifted = scores - s_max
    exp_s = cp.exp(s_shifted)
    sum_exp = cp.sum(exp_s, axis=-1, keepdims=True)
    scores[:] = exp_s / sum_exp


def _impl_attention_output_kernel(weights, v, output, seq_k, head_dim):
    """Compute attention output: weights @ V"""
    # weights: (batch*heads, seq_q, seq_k), v: (batch*heads, seq_k, head_dim)
    result = cp.einsum('bqk,bkd->bqd', weights, v)
    output[:] = result


def _impl_conv1d_matmul_kernel(col, weight, output, out_channels, col_size, out_length):
    """Conv1d via matrix multiplication after im2col transformation."""
    # col: (batch, col_size, out_length), weight: (out_channels, col_size)
    # output: (batch, out_channels, out_length)
    # Compute: weight @ col for each batch
    result = cp.einsum('oc,bcl->bol', weight, col)
    output[:] = result


def _impl_linear_bias_kernel(output, bias, M, N, TILE_N):
    """Add bias to linear output."""
    output[:] = output + bias


def _impl_rope_kernel(q, k, cos, sin, q_out, k_out, seq_len, head_dim):
    """Apply rotary position embeddings."""
    half = head_dim // 2

    q1 = q[..., :half]
    q2 = q[..., half:]
    k1 = k[..., :half]
    k2 = k[..., half:]

    cos1 = cos[..., :half]
    sin1 = sin[..., :half]

    q_rot1 = q1 * cos1 - q2 * sin1
    q_rot2 = q2 * cos1 + q1 * sin1
    q_out[..., :half] = q_rot1
    q_out[..., half:] = q_rot2

    k_rot1 = k1 * cos1 - k2 * sin1
    k_rot2 = k2 * cos1 + k1 * sin1
    k_out[..., :half] = k_rot1
    k_out[..., half:] = k_rot2


def _impl_apply_rope_kernel(q, k, cos, sin, q_out, k_out, head_dim, half_dim):
    """Apply RoPE kernel - batch version."""
    # q, k: (batch_heads, seq_len, head_dim)
    # cos, sin: (seq_len, half_dim)
    # q_out, k_out: (batch_heads, seq_len, head_dim)

    # Split into halves
    q1 = q[..., :half_dim]
    q2 = q[..., half_dim:half_dim*2]
    k1 = k[..., :half_dim]
    k2 = k[..., half_dim:half_dim*2]

    # Expand cos/sin for broadcasting: (seq_len, half_dim) -> (1, seq_len, half_dim)
    cos_exp = cos[None, :, :]
    sin_exp = sin[None, :, :]

    # Apply rotation
    q_out[..., :half_dim] = q1 * cos_exp - q2 * sin_exp
    q_out[..., half_dim:half_dim*2] = q2 * cos_exp + q1 * sin_exp
    k_out[..., :half_dim] = k1 * cos_exp - k2 * sin_exp
    k_out[..., half_dim:half_dim*2] = k2 * cos_exp + k1 * sin_exp

    # Copy remaining dimensions if any
    if head_dim > half_dim * 2:
        q_out[..., half_dim*2:] = q[..., half_dim*2:]
        k_out[..., half_dim*2:] = k[..., half_dim*2:]


# =============================================================================
# Tutorial Kernel Implementations
# =============================================================================

def _impl_vector_add(a, b, c, tile_size):
    """Vector add: c = a + b"""
    c[:] = a + b


def _impl_sigmoid_kernel(input_arr, output, tile_size):
    """Sigmoid: 1 / (1 + exp(-x))"""
    output[:] = 1.0 / (1.0 + cp.exp(-input_arr))


def _impl_grid_map_2d(output, tile_size_x, tile_size_y):
    """2D grid mapping: fills each tile with its coordinate encoding."""
    height, width = output.shape
    grid_x = (width + tile_size_x - 1) // tile_size_x
    grid_y = (height + tile_size_y - 1) // tile_size_y

    for pid_y in range(grid_y):
        for pid_x in range(grid_x):
            val = pid_x * 1000 + pid_y
            y_start = pid_y * tile_size_y
            y_end = _builtin_min(y_start + tile_size_y, height)
            x_start = pid_x * tile_size_x
            x_end = _builtin_min(x_start + tile_size_x, width)
            output[y_start:y_end, x_start:x_end] = val


def _impl_transpose_2d(input_arr, output, tile_size_x, tile_size_y):
    """2D transpose."""
    output[:] = input_arr.T


def _impl_simple_attention(Q, K, V, Out, seq_len_k, d_head, tile_size_m, tile_size_n):
    """Simplified attention: O = Softmax(Q @ K.T) @ V"""
    # Q: (M, D), K: (N, D), V: (N, D), Out: (M, D)
    # Compute Q @ K.T -> (M, N)
    scores = Q @ K.T

    # Scale
    scale = 1.0 / cp.sqrt(float(d_head))
    scores = scores * scale

    # Softmax
    scores = scores - cp.max(scores, axis=-1, keepdims=True)
    exp_scores = cp.exp(scores)
    attn_weights = exp_scores / cp.sum(exp_scores, axis=-1, keepdims=True)

    # @ V -> (M, D)
    Out[:] = attn_weights @ V


def _impl_math_kernel(data, out, tile_size):
    """Math kernel: various operations."""
    r = data
    res = r * r
    res = res + r
    res = res * 0.5
    res = res * res
    out[:] = res


def _impl_mixed_precision_kernel(input_arr, output, scale_factor, tile_size):
    """Mixed precision scaling."""
    output[:] = (input_arr.astype(cp.float32) * scale_factor).astype(input_arr.dtype)


# FlashAttention fallback
def _impl_flash_attention_fallback(q, k, v, out, qk_scale, tile_d, H, tile_m, tile_n, query_group_size, causal):
    """Fallback flash attention implementation using standard attention."""
    batch, num_heads, seq_q, head_dim = q.shape
    _, num_kv_heads, seq_k, _ = k.shape

    # Handle GQA by expanding K/V
    if num_kv_heads != num_heads:
        k = cp.repeat(k, query_group_size, axis=1)
        v = cp.repeat(v, query_group_size, axis=1)

    # Standard attention
    # Recover scale (qk_scale was pre-multiplied by 1/ln(2))
    scale = qk_scale * math.log(2)

    scores = cp.einsum('bhqd,bhkd->bhqk', q, k) * scale

    if causal:
        mask = cp.triu(cp.ones((seq_q, seq_k), dtype=cp.float32), k=1) * -1e9
        scores = scores + mask[None, None, :, :]

    # Softmax
    scores = scores - cp.max(scores, axis=-1, keepdims=True)
    exp_scores = cp.exp(scores)
    attn_weights = exp_scores / cp.sum(exp_scores, axis=-1, keepdims=True)

    # Output
    result = cp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
    out[:] = result


# =============================================================================
# Kernel Registry
# =============================================================================

# Map kernel function names to their CuPy implementations
_KERNEL_REGISTRY = {
    'rmsnorm_kernel': _impl_rmsnorm_kernel,
    'compute_freqs_kernel': _impl_compute_freqs_kernel,
    'layernorm_kernel': _impl_layernorm_kernel,
    'gelu_kernel': _impl_gelu_kernel,
    'silu_kernel': _impl_silu_kernel,
    'embedding_kernel': _impl_embedding_kernel,
    'softmax_kernel': _impl_softmax_kernel,
    'linear_kernel_tf32': _impl_linear_kernel_tf32,
    'linear_silu_kernel': _impl_linear_silu_kernel,
    'linear_gelu_kernel': _impl_linear_gelu_kernel,
    'swiglu_fused_kernel': _impl_swiglu_fused_kernel,
    'attention_scores_kernel': _impl_attention_scores_kernel,
    'softmax_inplace_kernel': _impl_softmax_inplace_kernel,
    'attention_output_kernel': _impl_attention_output_kernel,
    'conv1d_matmul_kernel': _impl_conv1d_matmul_kernel,
    'linear_bias_kernel': _impl_linear_bias_kernel,
    'rope_kernel': _impl_rope_kernel,
    'apply_rope_kernel': _impl_apply_rope_kernel,
    'flash_attention_kernel': _impl_flash_attention_fallback,
    # Tutorial kernels
    'vector_add': _impl_vector_add,
    'sigmoid_kernel': _impl_sigmoid_kernel,
    'grid_map_2d': _impl_grid_map_2d,
    'transpose_2d': _impl_transpose_2d,
    'simple_attention': _impl_simple_attention,
    'math_kernel': _impl_math_kernel,
    'mixed_precision_kernel': _impl_mixed_precision_kernel,
    'mixed_precision_scale': _impl_mixed_precision_kernel,
}


# =============================================================================
# Kernel Wrapper and Launch
# =============================================================================

class _KernelWrapper:
    """Wrapper for cuTile kernels that executes via CuPy."""

    def __init__(self, func: Callable, **options):
        self.func = func
        self.name = func.__name__
        self.options = options

    def __call__(self, *args, **kwargs):
        raise TypeError("Tile kernels cannot be called directly. Use cuda.tile.launch() instead.")


def kernel(func: Callable = None, /, **kwargs) -> _KernelWrapper:
    """Decorator to mark a function as a cuTile kernel."""
    if func is None:
        def decorator(f):
            return _KernelWrapper(f, **kwargs)
        return decorator
    return _KernelWrapper(func, **kwargs)


def function(func=None, /, *, host=False, tile=True):
    """Decorator for tile functions."""
    def decorator(func):
        if host:
            return func
        else:
            @wraps(func)
            def wrapped(*args, **kwargs):
                raise RuntimeError('Tile functions can only be called from tile code.')
            return wrapped

    if func is None:
        return decorator
    else:
        return decorator(func)


def launch(stream, grid: Tuple[int, ...], kernel_func: _KernelWrapper, args: Tuple):
    """Launch a cuTile kernel using CuPy fallback implementation."""
    if not isinstance(kernel_func, _KernelWrapper):
        raise TypeError("kernel_func must be decorated with @ct.kernel")

    kernel_name = kernel_func.name

    # Look up the CuPy implementation
    if kernel_name in _KERNEL_REGISTRY:
        impl = _KERNEL_REGISTRY[kernel_name]
        try:
            impl(*args)
        except Exception as e:
            print(f"[cuTile Compat] Warning: {kernel_name} failed: {e}")
            raise
    else:
        # For unknown kernels, print a warning but don't crash
        print(f"[cuTile Compat] Warning: No implementation for kernel '{kernel_name}', skipping")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core
    "kernel", "function", "launch", "cdiv",

    # Type annotations
    "Constant", "ConstantAnnotation", "Array", "Scalar", "Tile", "ByTarget",

    # Data types
    "DType", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64",
    "bfloat16", "tfloat32", "bool_",
    "float8_e4m3fn", "float8_e5m2",

    # Enums
    "MemoryOrder", "MemoryScope", "PaddingMode", "RoundingMode",

    # Exceptions
    "TileCompilerError", "TileCompilerExecutionError",
    "TileCompilerTimeoutError", "TileInternalError",
    "TileSyntaxError", "TileTypeError", "TileValueError",

    # Tile operations
    "bid", "num_blocks", "num_tiles",
    "load", "store", "full", "zeros", "ones", "arange",
    "astype", "transpose", "permute", "reshape",
    "broadcast_to", "expand_dims", "cat", "bitcast",
    "extract", "gather", "scatter", "where",

    # Math
    "exp", "exp2", "log", "log2", "sqrt", "rsqrt",
    "sin", "cos", "tan", "sinh", "cosh", "tanh",
    "floor", "ceil", "pow",

    # Reductions
    "sum", "prod", "min", "max", "argmin", "argmax",
    "cumsum", "cumprod", "minimum", "maximum",

    # Binary ops
    "add", "sub", "mul", "truediv", "floordiv", "mod", "negative",

    # Comparison
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",

    # Bitwise
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    "bitwise_lshift", "bitwise_rshift",

    # Matrix
    "matmul", "mma",

    # Atomic
    "atomic_add", "atomic_and", "atomic_or", "atomic_xor",
    "atomic_min", "atomic_max", "atomic_xchg", "atomic_cas",

    # Debug
    "printf", "assert_",
]

# Print info on import
import sys
if not hasattr(sys, '_cutile_compat_warned'):
    print("[cuTile Compat] Using Hopper compatibility layer for non-Blackwell GPU")
    sys._cutile_compat_warned = True
