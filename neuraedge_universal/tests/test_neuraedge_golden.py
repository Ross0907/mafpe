from __future__ import annotations

import unittest

import numpy as np

from neuraedge_golden import (
    Dataflow,
    VectorOperation,
    active_dot8_indices,
    active_outer_indices,
    alpha_xnor_similarity,
    binary_weighted_gemm,
    choose_dataflow,
    decode_uint8_as_int8,
    direct_centered_gemm,
    encode_int8_as_uint8,
    gemm_u8_zero_point,
    integer_layer_norm,
    linear_backward_tile,
    lif_step,
    log_int_softmax_q15,
    packed_mac2_u8,
    peak_dense_gops,
    quantize_flexible_scale,
    round_shift_signed,
    saturate_signed,
    sgd_momentum_step,
    ternary_spike,
    ternary_weighted_gemm,
    vector_alu,
)


class GoldenModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(20260711)

    def test_signed_encoding_round_trip(self) -> None:
        values = np.arange(-128, 128, dtype=np.int16)
        np.testing.assert_array_equal(decode_uint8_as_int8(encode_int8_as_uint8(values)), values)

    def test_packed_mac_boundaries(self) -> None:
        for activation in (0, 1, 127, 128, 254, 255):
            for low in (0, 1, 127, 128, 254, 255):
                for high in (0, 1, 127, 128, 254, 255):
                    got_low, got_high = packed_mac2_u8(activation, low, high)
                    self.assertEqual(got_low, activation * low)
                    self.assertEqual(got_high, activation * high)

    def test_zero_point_correction_matches_direct_expression(self) -> None:
        for _ in range(100):
            rows = int(self.rng.integers(1, 9))
            columns = int(self.rng.integers(1, 33))
            reduction = int(self.rng.integers(1, 129))
            a = self.rng.integers(0, 256, size=(rows, reduction), dtype=np.uint8)
            b = self.rng.integers(0, 256, size=(reduction, columns), dtype=np.uint8)
            za = int(self.rng.integers(0, 256))
            zb = int(self.rng.integers(0, 256))
            bias = self.rng.integers(-(1 << 20), 1 << 20, size=columns, dtype=np.int64)
            corrected = gemm_u8_zero_point(a, b, za, zb, bias)
            direct = direct_centered_gemm(a, b, za, zb, bias)
            np.testing.assert_array_equal(corrected, direct)

    def test_sparse_index_compaction(self) -> None:
        zero = 128
        a = np.full((3, 19), zero, dtype=np.uint8)
        a[0, 1] = 129
        a[2, 9] = 127
        a[1, 18] = 0
        np.testing.assert_array_equal(active_outer_indices(a, zero), np.array([1, 9, 18]))
        np.testing.assert_array_equal(active_dot8_indices(a[0], zero), np.array([0]))
        np.testing.assert_array_equal(active_dot8_indices(a[2], zero), np.array([1]))

    def test_binary_and_ternary_products(self) -> None:
        weights = self.rng.integers(-127, 128, size=(17, 11), dtype=np.int16)
        binary = self.rng.integers(0, 2, size=(5, 17), dtype=np.int8)
        ternary = self.rng.integers(-1, 2, size=(5, 17), dtype=np.int8)
        np.testing.assert_array_equal(binary_weighted_gemm(binary, weights), binary.astype(np.int64) @ weights)
        np.testing.assert_array_equal(ternary_weighted_gemm(ternary, weights), ternary.astype(np.int64) @ weights)

    def test_alpha_xnor_special_cases(self) -> None:
        q = np.array([[1, 0, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)
        k = np.array([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
        dot_q8 = alpha_xnor_similarity(q, k, 0)
        xnor_q8 = alpha_xnor_similarity(q, k, 256)
        np.testing.assert_array_equal(dot_q8 // 256, q.astype(np.int64) @ k.astype(np.int64).T)
        expected_xnor = np.array([[3, 2], [3, 4]], dtype=np.int64)
        np.testing.assert_array_equal(xnor_q8 // 256, expected_xnor)

    def test_log_integer_softmax_properties(self) -> None:
        scores = np.array([91, 83, 50, -30], dtype=np.int64)
        probabilities = log_int_softmax_q15(scores, difference_shift=3)
        translated = log_int_softmax_q15(scores + 123456, difference_shift=3)
        np.testing.assert_array_equal(probabilities, translated)
        self.assertGreaterEqual(int(probabilities[0]), int(probabilities[1]))
        self.assertGreaterEqual(int(probabilities[1]), int(probabilities[2]))
        self.assertLessEqual(abs(int(probabilities.sum()) - (1 << 15)), 2)

    def test_integer_layer_norm_constant_and_nonconstant(self) -> None:
        gamma = np.full(4, 1 << 15, dtype=np.int64)
        beta = np.array([1, 2, 3, 4], dtype=np.int64)
        np.testing.assert_array_equal(integer_layer_norm(np.full(4, 12), gamma, beta), beta)
        output = integer_layer_norm(np.array([-9, -3, 3, 9]), gamma, np.zeros(4, dtype=np.int64))
        self.assertEqual(int(output.sum()), 0)
        self.assertTrue(np.all(np.diff(output) >= 0))

    def test_lif_and_ternary_firing(self) -> None:
        next_membrane, spikes = lif_step(np.array([0, 8, 20]), np.array([4, 8, 1]), 2, 12)
        np.testing.assert_array_equal(spikes, np.array([0, 1, 1], dtype=np.uint8))
        np.testing.assert_array_equal(next_membrane, np.array([4, 0, 0]))
        np.testing.assert_array_equal(
            ternary_spike(np.array([-8, -7, 0, 7, 8]), 8),
            np.array([-1, 0, 0, 0, 1], dtype=np.int8),
        )

    def test_dataflow_selection_and_peak_rate(self) -> None:
        wide = choose_dataflow(rows=8, columns=768, reduction=384)
        tall = choose_dataflow(rows=4096, columns=16, reduction=144)
        self.assertEqual(wide.mode, Dataflow.INPUT_BROADCAST)
        self.assertEqual(wide.input_broadcast_bytes, 297984)
        self.assertEqual(wide.weight_broadcast_bytes, 442368)
        self.assertEqual(tall.mode, Dataflow.WEIGHT_BROADCAST)
        self.assertEqual(tall.input_broadcast_bytes, 1769472)
        self.assertEqual(tall.weight_broadcast_bytes, 592128)
        self.assertAlmostEqual(peak_dense_gops(200e6, 64), 51.2)

    def test_signed_rounding_and_saturation(self) -> None:
        values = np.array([-9, -8, -7, -1, 0, 1, 7, 8, 9], dtype=np.int64)
        np.testing.assert_array_equal(
            round_shift_signed(values, 3),
            np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            saturate_signed(np.array([-40000, -32768, 32767, 40000]), 16),
            np.array([-32768, -32768, 32767, 32767]),
        )

    def test_linear_backward_matches_independent_scalar_reference(self) -> None:
        def rounded(value: int, shift: int) -> int:
            if shift == 0:
                return value
            magnitude = abs(value)
            result = (magnitude + (1 << (shift - 1))) >> shift
            return -result if value < 0 else result

        def clipped(value: int, bits: int) -> int:
            return max(-(1 << (bits - 1)), min((1 << (bits - 1)) - 1, value))

        for _ in range(100):
            rows = int(self.rng.integers(1, 9))
            inputs = int(self.rng.integers(1, 9))
            outputs = int(self.rng.integers(1, 9))
            dx_shift = int(self.rng.integers(0, 13))
            dw_shift = int(self.rng.integers(0, 13))
            x = self.rng.integers(-128, 128, size=(rows, inputs), dtype=np.int64)
            w = self.rng.integers(-32768, 32768, size=(inputs, outputs), dtype=np.int64)
            dy = self.rng.integers(-32768, 32768, size=(rows, outputs), dtype=np.int64)
            expected_dx = np.zeros((rows, inputs), dtype=np.int64)
            expected_dw = np.zeros((inputs, outputs), dtype=np.int64)
            for row in range(rows):
                for input_index in range(inputs):
                    total = sum(
                        int(dy[row, output_index]) * int(w[input_index, output_index])
                        for output_index in range(outputs)
                    )
                    expected_dx[row, input_index] = clipped(rounded(total, dx_shift), 32)
            for input_index in range(inputs):
                for output_index in range(outputs):
                    total = sum(
                        int(x[row, input_index]) * int(dy[row, output_index])
                        for row in range(rows)
                    )
                    expected_dw[input_index, output_index] = clipped(rounded(total, dw_shift), 24)
            got_dx, got_dw = linear_backward_tile(
                x,
                w,
                dy,
                input_gradient_shift=dx_shift,
                weight_gradient_shift=dw_shift,
            )
            np.testing.assert_array_equal(got_dx, expected_dx)
            np.testing.assert_array_equal(got_dw, expected_dw)

        x = np.full((8, 8), 127, dtype=np.int64)
        w = np.full((8, 8), -32768, dtype=np.int64)
        dy = np.full((8, 8), -32768, dtype=np.int64)
        got_dx, got_dw = linear_backward_tile(x, w, dy)
        np.testing.assert_array_equal(got_dx, np.full((8, 8), (1 << 31) - 1))
        np.testing.assert_array_equal(got_dw, np.full((8, 8), -(1 << 23)))

    def test_sgd_momentum_update(self) -> None:
        for _ in range(100):
            weights = self.rng.integers(-32768, 32768, size=8, dtype=np.int64)
            gradients = self.rng.integers(-(1 << 20), 1 << 20, size=8, dtype=np.int64)
            velocities = self.rng.integers(-(1 << 20), 1 << 20, size=8, dtype=np.int64)
            next_weight, next_velocity = sgd_momentum_step(
                weights, gradients, velocities, 655, 58982, 66
            )
            self.assertTrue(np.all(next_weight >= -32768))
            self.assertTrue(np.all(next_weight <= 32767))
            self.assertTrue(np.all(next_velocity >= -(1 << 23)))
            self.assertTrue(np.all(next_velocity <= (1 << 23) - 1))

    def test_sgd_weight_decay_is_decoupled_from_velocity(self) -> None:
        next_weight, next_velocity = sgd_momentum_step(
            np.full(8, 1000, dtype=np.int64),
            np.zeros(8, dtype=np.int64),
            np.full(8, 1000, dtype=np.int64),
            65536,
            65536,
            65536,
        )
        np.testing.assert_array_equal(next_velocity, np.full(8, 1000))
        np.testing.assert_array_equal(next_weight, np.full(8, -1000))

    def test_vector_alu_operations(self) -> None:
        a = np.array([-(1 << 31), -100, -8, -1, 0, 7, 100, (1 << 31) - 1])
        b = np.array([-1, 300, 8, -2, 1, 9, -300, 1])
        added = vector_alu(VectorOperation.ADD_SAT, a, b)
        self.assertEqual(int(added[0]), -(1 << 31))
        self.assertEqual(int(added[-1]), (1 << 31) - 1)
        np.testing.assert_array_equal(
            vector_alu(VectorOperation.TERNARY_SPIKE, a, b, threshold=8),
            np.array([-1, -1, -1, 0, 0, 0, 1, 1]),
        )
        requantized = vector_alu(
            VectorOperation.REQUANTIZE,
            a,
            b,
            multiplier_q15=16384,
            right_shift=1,
            output_zero_point=3,
            clamp_minimum=-128,
            clamp_maximum=127,
        )
        self.assertTrue(np.all(requantized >= -128))
        self.assertTrue(np.all(requantized <= 127))

        multiplier, shift, represented = quantize_flexible_scale(51.03228)
        self.assertGreater(multiplier, 0)
        self.assertLess(shift, 0)
        self.assertAlmostEqual(represented, 51.03228, delta=0.002)
        flexible = vector_alu(
            VectorOperation.FLEX_REQUANTIZE,
            np.array([1, -1, 100, -100], dtype=np.int64),
            np.zeros(4, dtype=np.int64),
            multiplier_q15=multiplier,
            right_shift=shift,
        )
        np.testing.assert_allclose(
            flexible,
            np.array([51, -51, 5103, -5103], dtype=np.int64),
            atol=1,
        )
        selector = np.array([0, 1, -1, 0], dtype=np.int64)
        payload = np.array([10, 20, 30, 40], dtype=np.int64)
        np.testing.assert_array_equal(
            vector_alu(VectorOperation.SELECT_ZERO, selector, payload),
            np.array([10, 0, 0, 40]),
        )
        np.testing.assert_array_equal(
            vector_alu(VectorOperation.SELECT_NONZERO, selector, payload),
            np.array([0, 20, 30, 0]),
        )
        np.testing.assert_array_equal(
            vector_alu(
                VectorOperation.COMPARE_GE,
                np.array([1, 2, 2, -1]),
                np.array([2, 2, 1, -1]),
            ),
            np.array([0, 1, 1, 1]),
        )


if __name__ == "__main__":
    unittest.main()
