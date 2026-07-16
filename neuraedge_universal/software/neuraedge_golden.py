"""Independent integer reference model for the NeuraEdge universal accelerator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import ceil, frexp, isqrt

import numpy as np


class Dataflow(IntEnum):
    INPUT_BROADCAST = 0
    WEIGHT_BROADCAST = 1


class ComputeMode(IntEnum):
    DENSE_OUTER = 0
    DENSE_DOT8 = 1
    BINARY_WEIGHTED = 2
    TERNARY_WEIGHTED = 3


class VectorOperation(IntEnum):
    ADD_SAT = 0
    RELU = 1
    REQUANTIZE = 2
    BINARY_SPIKE = 3
    TERNARY_SPIKE = 4
    SURROGATE_GRADIENT = 5
    MAXIMUM = 6
    MULTIPLY_Q15 = 7
    CLAMP = 8
    FLEX_REQUANTIZE = 9
    SELECT_ZERO = 10
    SELECT_NONZERO = 11
    COMPARE_GE = 12


@dataclass(frozen=True)
class DataflowDecision:
    mode: Dataflow
    input_broadcast_bytes: int
    weight_broadcast_bytes: int


def encode_int8_as_uint8(values: np.ndarray) -> np.ndarray:
    """Map signed INT8 bit patterns to the accelerator's unsigned arithmetic domain."""
    signed = np.asarray(values, dtype=np.int16)
    if np.any(signed < -128) or np.any(signed > 127):
        raise ValueError("signed INT8 values must be in [-128, 127]")
    return (signed + 128).astype(np.uint8)


def decode_uint8_as_int8(values: np.ndarray) -> np.ndarray:
    """Undo encode_int8_as_uint8 without relying on implementation-defined casts."""
    unsigned = np.asarray(values, dtype=np.int16)
    if np.any(unsigned < 0) or np.any(unsigned > 255):
        raise ValueError("encoded values must be in [0, 255]")
    return (unsigned - 128).astype(np.int16)


def packed_mac2_u8(activation: int, weight_lo: int, weight_hi: int) -> tuple[int, int]:
    """Bit-accurate model of neuraedge_mac2_u8, including its guard-bit packing."""
    for name, value in (("activation", activation), ("weight_lo", weight_lo), ("weight_hi", weight_hi)):
        if not 0 <= value <= 255:
            raise ValueError(f"{name} must be an unsigned byte")
    packed_weight = weight_lo | (weight_hi << 17)
    packed_product = packed_weight * activation
    return packed_product & 0xFFFF, (packed_product >> 17) & 0xFFFF


def gemm_u8_zero_point(
    activations: np.ndarray,
    weights: np.ndarray,
    activation_zero_point: int,
    weight_zero_point: int,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Compute (A-Za)(B-Zb)+bias using the accelerator's raw-sum correction."""
    a = np.asarray(activations, dtype=np.int64)
    b = np.asarray(weights, dtype=np.int64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if np.any(a < 0) or np.any(a > 255) or np.any(b < 0) or np.any(b > 255):
        raise ValueError("accelerator GEMM operands must be unsigned bytes")
    if not 0 <= activation_zero_point <= 255 or not 0 <= weight_zero_point <= 255:
        raise ValueError("zero points must be unsigned bytes")

    reduction = a.shape[1]
    raw = a @ b
    row_sum = a.sum(axis=1, keepdims=True)
    column_sum = b.sum(axis=0, keepdims=True)
    corrected = (
        raw
        - weight_zero_point * row_sum
        - activation_zero_point * column_sum
        + reduction * activation_zero_point * weight_zero_point
    )
    if bias is not None:
        bias_array = np.asarray(bias, dtype=np.int64).reshape(1, -1)
        if bias_array.shape[1] != b.shape[1]:
            raise ValueError("bias length must equal output columns")
        corrected = corrected + bias_array
    return corrected


def direct_centered_gemm(
    activations: np.ndarray,
    weights: np.ndarray,
    activation_zero_point: int,
    weight_zero_point: int,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Simple centered expression used only as an independent algebra oracle."""
    a = np.asarray(activations, dtype=np.int64) - activation_zero_point
    b = np.asarray(weights, dtype=np.int64) - weight_zero_point
    result = a @ b
    if bias is not None:
        result = result + np.asarray(bias, dtype=np.int64).reshape(1, -1)
    return result


def active_outer_indices(activations: np.ndarray, zero_point: int) -> np.ndarray:
    """Return K indices where at least one row contains a non-zero activation."""
    a = np.asarray(activations)
    if a.ndim != 2:
        raise ValueError("outer-mode activations must have shape [M,K]")
    return np.flatnonzero(np.any(a != zero_point, axis=0)).astype(np.int64)


def active_dot8_indices(activation_row: np.ndarray, zero_point: int) -> np.ndarray:
    """Return eight-element K chunks containing at least one non-zero activation."""
    row = np.asarray(activation_row).reshape(-1)
    padded_length = ceil(row.size / 8) * 8
    padded = np.full(padded_length, zero_point, dtype=row.dtype)
    padded[: row.size] = row
    chunks = padded.reshape(-1, 8)
    return np.flatnonzero(np.any(chunks != zero_point, axis=1)).astype(np.int64)


def binary_weighted_gemm(spikes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Addition-only matrix product for binary activations in {0,1}."""
    x = np.asarray(spikes, dtype=np.int64)
    w = np.asarray(weights, dtype=np.int64)
    if np.any((x != 0) & (x != 1)):
        raise ValueError("binary activations must contain only 0 and 1")
    return x @ w


def ternary_weighted_gemm(spikes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Addition/subtraction-only matrix product for activations in {-1,0,1}."""
    x = np.asarray(spikes, dtype=np.int64)
    w = np.asarray(weights, dtype=np.int64)
    if np.any((x < -1) | (x > 1)):
        raise ValueError("ternary activations must be in {-1,0,1}")
    return x @ w


def alpha_xnor_similarity(query: np.ndarray, key: np.ndarray, alpha_q8: int) -> np.ndarray:
    """Equation 18 of Xiao et al. CVPR 2025, represented in unsigned Q8."""
    q = np.asarray(query, dtype=np.uint8)
    k = np.asarray(key, dtype=np.uint8)
    if q.ndim != 2 or k.ndim != 2 or q.shape[1] != k.shape[1]:
        raise ValueError("expected query[M,D] and key[N,D]")
    if np.any(q > 1) or np.any(k > 1):
        raise ValueError("alpha-XNOR inputs must be binary")
    if not 0 <= alpha_q8 <= 256:
        raise ValueError("alpha_q8 must represent a value in [0,1]")
    spike_matches = q.astype(np.int64) @ k.astype(np.int64).T
    zero_matches = (1 - q.astype(np.int64)) @ (1 - k.astype(np.int64)).T
    return spike_matches * 256 + zero_matches * alpha_q8


def log_int_softmax_q15(scores: np.ndarray, difference_shift: int) -> np.ndarray:
    """Power-of-two softmax with the Q31 reciprocal used by the RTL datapath."""
    x = np.asarray(scores, dtype=np.int64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("scores must be a non-empty vector")
    if not 0 <= difference_shift <= 31:
        raise ValueError("difference_shift must be in [0,31]")
    differences = x.max() - x
    exponents = np.minimum(differences >> difference_shift, 15).astype(np.int64)
    unnormalized = (1 << (15 - exponents)).astype(np.int64)
    denominator = int(unnormalized.sum())
    reciprocal_q31 = ((1 << 31) - 1) // denominator
    return round_shift_signed(unnormalized * reciprocal_q31, 16)


def integer_layer_norm(
    values: np.ndarray,
    gamma_q15: np.ndarray,
    beta: np.ndarray,
    epsilon: int = 1,
) -> np.ndarray:
    """I-ViT-style integer LayerNorm using sqrt and one shared Q31 reciprocal."""
    x = np.asarray(values, dtype=np.int64).reshape(-1)
    gamma = np.asarray(gamma_q15, dtype=np.int64).reshape(-1)
    offset = np.asarray(beta, dtype=np.int64).reshape(-1)
    if x.size == 0 or gamma.size != x.size or offset.size != x.size:
        raise ValueError("values, gamma, and beta must be equally sized non-empty vectors")
    mean = int(x.sum() // x.size)
    centered = x - mean
    variance = int((centered * centered).sum() // x.size)
    sigma = max(1, isqrt(variance + epsilon))
    reciprocal_q31 = ((1 << 31) - 1) // sigma
    normalized = round_shift_signed(centered * reciprocal_q31, 31)
    return round_shift_signed(normalized * gamma, 15) + offset


def lif_step(
    membrane: np.ndarray,
    current: np.ndarray,
    leak_shift: int,
    threshold: int,
    reset_value: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """One deterministic leaky-integrate-and-fire step."""
    v = np.asarray(membrane, dtype=np.int64)
    i = np.asarray(current, dtype=np.int64)
    integrated = v - (v >> leak_shift) + i
    spikes = integrated >= threshold
    next_membrane = np.where(spikes, reset_value, integrated)
    return next_membrane.astype(np.int64), spikes.astype(np.uint8)


def ternary_spike(values: np.ndarray, threshold: int) -> np.ndarray:
    """Symmetric ternary firing rule used by the addition-only datapath."""
    x = np.asarray(values, dtype=np.int64)
    return np.where(x >= threshold, 1, np.where(x <= -threshold, -1, 0)).astype(np.int8)


def saturate_signed(values: np.ndarray | int, bits: int) -> np.ndarray:
    """Clamp integers to the exact range of a signed two's-complement field."""
    if bits < 2 or bits > 63:
        raise ValueError("bits must be in [2,63]")
    array = np.asarray(values, dtype=np.int64)
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    return np.clip(array, minimum, maximum).astype(np.int64)


def round_shift_signed(values: np.ndarray | int, shift: int) -> np.ndarray:
    """Round signed integers to nearest, with exact half cases away from zero."""
    if not 0 <= shift <= 62:
        raise ValueError("shift must be in [0,62]")
    array = np.asarray(values, dtype=np.int64)
    if shift == 0:
        return array.copy()
    magnitude = np.abs(array)
    rounded = (magnitude + (1 << (shift - 1))) >> shift
    return np.where(array < 0, -rounded, rounded).astype(np.int64)


def linear_backward_tile(
    activations: np.ndarray,
    weights: np.ndarray,
    output_gradients: np.ndarray,
    *,
    input_gradient_shift: int = 0,
    weight_gradient_shift: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bit-exact reference for the signed RTL linear backward tile."""

    x = np.asarray(activations, dtype=np.int64)
    w = np.asarray(weights, dtype=np.int64)
    dy = np.asarray(output_gradients, dtype=np.int64)
    if x.ndim != 2 or w.ndim != 2 or dy.ndim != 2:
        raise ValueError("linear backward operands must be two-dimensional")
    rows, input_features = x.shape
    weight_inputs, output_features = w.shape
    if weight_inputs != input_features or dy.shape != (rows, output_features):
        raise ValueError("linear backward matrix dimensions do not align")
    if not 1 <= rows <= 8 or not 1 <= input_features <= 8 or not 1 <= output_features <= 8:
        raise ValueError("linear backward tile dimensions must be in [1,8]")
    if np.any(x < -(1 << 15)) or np.any(x > (1 << 15) - 1):
        raise ValueError("linear backward activations must fit signed 16-bit")
    if np.any(w < -(1 << 15)) or np.any(w > (1 << 15) - 1):
        raise ValueError("linear backward weights must fit signed 16-bit")
    if np.any(dy < -(1 << 15)) or np.any(dy > (1 << 15) - 1):
        raise ValueError("linear backward output gradients must fit signed 16-bit")
    if not 0 <= input_gradient_shift <= 31 or not 0 <= weight_gradient_shift <= 31:
        raise ValueError("linear backward shifts must be in [0,31]")

    input_gradient = round_shift_signed(dy @ w.T, input_gradient_shift)
    weight_gradient = round_shift_signed(x.T @ dy, weight_gradient_shift)
    return saturate_signed(input_gradient, 32), saturate_signed(weight_gradient, 24)


def quantize_flexible_scale(scale: float) -> tuple[int, int, float]:
    """Encode a signed scale as Q15 mantissa and signed binary-point shift.

    The flexible vector operation evaluates ``x * multiplier / 2**(15+shift)``.
    Its signed five-bit shift covers [-15, 15], giving useful non-zero scale
    magnitudes from approximately 2^-16 through 2^15.
    """

    if not np.isfinite(scale):
        raise ValueError("scale must be finite")
    if scale == 0.0:
        return 0, 0, 0.0
    mantissa, exponent = frexp(scale)
    multiplier = int(round(mantissa * (1 << 15)))
    if multiplier == (1 << 15):
        multiplier //= 2
        exponent += 1
    if multiplier < -(1 << 15) or multiplier > (1 << 15) - 1:
        raise ValueError("scale mantissa does not fit signed Q15")
    shift = -exponent
    if not -15 <= shift <= 15:
        raise ValueError("scale is outside the flexible requantizer range")
    represented = multiplier / float(1 << (15 + shift))
    return multiplier, shift, represented


def sgd_momentum_step(
    weights: np.ndarray,
    gradients: np.ndarray,
    velocities: np.ndarray,
    learning_rate_q16: int,
    momentum_q16: int,
    weight_decay_q16: int,
    update_right_shift: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Bit-exact SGDW update used by the fabric optimizer lanes."""
    w = np.asarray(weights, dtype=np.int64)
    g = np.asarray(gradients, dtype=np.int64)
    v = np.asarray(velocities, dtype=np.int64)
    if w.shape != g.shape or w.shape != v.shape:
        raise ValueError("weights, gradients, and velocities must have equal shapes")
    if not 0 <= learning_rate_q16 <= 131071:
        raise ValueError("learning_rate_q16 must be in [0,131071]")
    if not 0 <= momentum_q16 <= 65536 or not 0 <= weight_decay_q16 <= 65536:
        raise ValueError("momentum and weight decay must be unsigned Q16 values")
    if not 0 <= update_right_shift <= 31:
        raise ValueError("update_right_shift must be in [0,31]")
    momentum_term = round_shift_signed(v * momentum_q16, 16)
    decay_term = round_shift_signed(w * weight_decay_q16, 16)
    next_velocity = saturate_signed(momentum_term + g, 24)
    update_direction = saturate_signed(next_velocity + decay_term, 24)
    update = round_shift_signed(
        update_direction * learning_rate_q16, update_right_shift)
    next_weight = saturate_signed(w - update, 16)
    return next_weight, next_velocity


def vector_alu(
    operation: VectorOperation,
    operand_a: np.ndarray,
    operand_b: np.ndarray,
    *,
    multiplier_q15: int = 0,
    right_shift: int = 0,
    output_zero_point: int = 0,
    threshold: int = 0,
    surrogate_window: int = 0,
    clamp_minimum: int = -(1 << 31),
    clamp_maximum: int = (1 << 31) - 1,
) -> np.ndarray:
    """Reference for every operation in the eight-lane integer vector ALU."""
    a = np.asarray(operand_a, dtype=np.int64)
    b = np.asarray(operand_b, dtype=np.int64)
    if a.shape != b.shape:
        raise ValueError("operand vectors must have equal shapes")
    if operation == VectorOperation.FLEX_REQUANTIZE:
        if not -15 <= right_shift <= 15:
            raise ValueError("flexible binary-point shift must be in [-15,15]")
    elif not 0 <= right_shift <= 31:
        raise ValueError("right_shift must be in [0,31]")
    if clamp_minimum > clamp_maximum:
        raise ValueError("clamp_minimum must not exceed clamp_maximum")

    if operation == VectorOperation.ADD_SAT:
        output = saturate_signed(a + b, 32)
    elif operation == VectorOperation.RELU:
        output = np.maximum(a, 0)
    elif operation == VectorOperation.REQUANTIZE:
        output = round_shift_signed(a * multiplier_q15, 15 + right_shift)
        output = np.clip(output + output_zero_point, clamp_minimum, clamp_maximum)
    elif operation == VectorOperation.FLEX_REQUANTIZE:
        output = round_shift_signed(a * multiplier_q15, 15 + right_shift)
        output = np.clip(output + output_zero_point, clamp_minimum, clamp_maximum)
    elif operation == VectorOperation.BINARY_SPIKE:
        output = (a >= threshold).astype(np.int64)
    elif operation == VectorOperation.TERNARY_SPIKE:
        output = np.where(a >= threshold, 1, np.where(a <= -threshold, -1, 0))
    elif operation == VectorOperation.SURROGATE_GRADIENT:
        output = np.where(np.abs(a - threshold) <= surrogate_window, b, 0)
    elif operation == VectorOperation.MAXIMUM:
        output = np.maximum(a, b)
    elif operation == VectorOperation.MULTIPLY_Q15:
        output = saturate_signed(round_shift_signed(a * b, 15), 32)
    elif operation == VectorOperation.CLAMP:
        output = np.clip(a, clamp_minimum, clamp_maximum)
    elif operation == VectorOperation.SELECT_ZERO:
        output = np.where(a == 0, b, 0)
    elif operation == VectorOperation.SELECT_NONZERO:
        output = np.where(a != 0, b, 0)
    elif operation == VectorOperation.COMPARE_GE:
        output = (a >= b).astype(np.int64)
    else:
        raise ValueError(f"unsupported vector operation {operation}")
    return np.asarray(output, dtype=np.int64)


def choose_dataflow(
    rows: int,
    columns: int,
    reduction: int,
    row_tile: int = 8,
    column_tile: int = 16,
) -> DataflowDecision:
    """Choose the traversal with fewer estimated operand bytes."""
    if min(rows, columns, reduction, row_tile, column_tile) <= 0:
        raise ValueError("matrix and tile dimensions must be positive")
    row_tiles = ceil(rows / row_tile)
    column_tiles = ceil(columns / column_tile)
    input_tile_bytes = row_tile * reduction
    weight_tile_bytes = column_tile * reduction
    input_broadcast_bytes = row_tiles * input_tile_bytes + row_tiles * column_tiles * weight_tile_bytes
    weight_broadcast_bytes = column_tiles * weight_tile_bytes + column_tiles * row_tiles * input_tile_bytes
    mode = (
        Dataflow.INPUT_BROADCAST
        if input_broadcast_bytes <= weight_broadcast_bytes
        else Dataflow.WEIGHT_BROADCAST
    )
    return DataflowDecision(mode, input_broadcast_bytes, weight_broadcast_bytes)


def peak_dense_gops(clock_hz: float, packed_dsp_pairs: int) -> float:
    """Two operations per MAC and two MACs per packed DSP pair."""
    return clock_hz * packed_dsp_pairs * 2 * 2 / 1e9
