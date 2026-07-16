import unittest

from neuraedge_protocol import (
    Command128,
    ProgramRecord,
    StreamBeat,
    LinearGradientMode,
    LutSuboperation,
    Opcode,
    alpha_xnor_command,
    clear_status_command,
    layernorm_command,
    linear_backward_command,
    lif_command,
    log_softmax_command,
    lut_command,
    pack_activation_beats,
    pack_alpha_beats,
    pack_dot8_weight_beats,
    pack_layernorm_beats,
    pack_linear_backward_beats,
    pack_lif_beats,
    pack_lut_eval_beat,
    pack_lut_load_beats,
    pack_optimizer_beats,
    pack_outer_weight_beats,
    pack_softmax_beats,
    pack_vector_beats,
    pack_descriptor_program,
    sgd_command,
    tensor_command,
    unpack_alpha_result,
    unpack_layernorm_result,
    unpack_linear_backward_result,
    unpack_lif_result,
    unpack_lif_trace,
    unpack_lut_result,
    unpack_optimizer_result,
    unpack_softmax_result,
    unpack_tensor_result,
    unpack_vector_result,
    vector_command,
)


class ProtocolTests(unittest.TestCase):
    def test_command_words_round_trip(self):
        command = Command128(0xFEDCBA98765432100123456789ABCDEF)
        self.assertEqual(command.words_le, (0x89ABCDEF, 0x01234567, 0x76543210, 0xFEDCBA98))
        self.assertEqual(Command128.from_words_le(command.words_le), command)
        self.assertEqual(clear_status_command().opcode, Opcode.CLEAR_STATUS)

    def test_tensor_command_fields_and_limits(self):
        command = tensor_command(
            Opcode.COMPUTE_DENSE_OUTER,
            rows=8,
            columns=16,
            issue_count=2048,
            activation_zero_point=128,
            weight_zero_point=129,
        ).value
        self.assertEqual(command & 0xF, Opcode.COMPUTE_DENSE_OUTER)
        self.assertEqual((command >> 4) & 0xF, 8)
        self.assertEqual((command >> 8) & 0x3F, 16)
        self.assertEqual((command >> 14) & 0xFFF, 2048)
        self.assertEqual((command >> 26) & 0xFF, 128)
        self.assertEqual((command >> 34) & 0xFF, 129)
        with self.assertRaises(ValueError):
            tensor_command(Opcode.COMPUTE_DENSE_DOT8, rows=2, columns=16, issue_count=1)
        with self.assertRaises(ValueError):
            tensor_command(Opcode.LOAD_WEIGHTS_DOT8, issue_count=257)

    def test_non_tensor_command_fields(self):
        alpha = alpha_xnor_command(256, 127).value
        self.assertEqual((alpha >> 42) & 0x1FF, 256)
        self.assertEqual((alpha >> 51) & 0x1FF, 127)
        vector = vector_command(8, multiplier_q15=-3, right_shift=31,
                                output_zero_point=-8, threshold=-123,
                                surrogate_window=456).value
        self.assertEqual((vector >> 4) & 0xF, 8)
        self.assertEqual((vector >> 8) & 0xFFFF, 0xFFFD)
        self.assertEqual((vector >> 24) & 0x1F, 31)
        self.assertEqual((vector >> 29) & 0xFFFF, 0xFFF8)
        self.assertEqual((vector >> 45) & 0xFFFFFFFF, (-123) & 0xFFFFFFFF)
        self.assertEqual((vector >> 77) & 0xFFFFFFFF, 456)
        flexible = vector_command(9, multiplier_q15=24576, right_shift=-6).value
        self.assertEqual((flexible >> 4) & 0xF, 9)
        self.assertEqual((flexible >> 24) & 0x1F, (-6) & 0x1F)
        self.assertEqual((layernorm_command(2048, 17).value >> 4) & 0xFFF, 2048)
        self.assertEqual((log_softmax_command(256, 15).value >> 4) & 0x1FF, 256)
        lif = lif_command(4, -2, -3, clear_state=True, trace_state=True).value
        self.assertEqual((lif >> 57) & 1, 1)
        self.assertEqual((lif >> 58) & 1, 1)
        optimizer = sgd_command(
            131071, 65536, 32768, update_right_shift=29).value
        self.assertEqual((optimizer >> 4) & 0x3FFFF, 131071)
        self.assertEqual((optimizer >> 22) & 0x1FFFF, 65536)
        self.assertEqual((optimizer >> 39) & 0x1FFFF, 32768)
        self.assertEqual((optimizer >> 56) & 0x1F, 29)
        self.assertEqual(lut_command(LutSuboperation.EVALUATE, 8).opcode, Opcode.ACTIVATION_LUT)

        backward = linear_backward_command(
            8,
            7,
            6,
            input_gradient_shift=31,
            weight_gradient_shift=17,
            mode=LinearGradientMode.WEIGHT,
            accumulate=True,
            suppress_output=True,
        ).value
        self.assertEqual(backward & 0xF, Opcode.TRAIN_LINEAR_BACKWARD)
        self.assertEqual((backward >> 4) & 0xF, 8)
        self.assertEqual((backward >> 8) & 0xF, 7)
        self.assertEqual((backward >> 12) & 0xF, 6)
        self.assertEqual((backward >> 16) & 0x1F, 31)
        self.assertEqual((backward >> 21) & 0x1F, 17)
        self.assertEqual((backward >> 26) & 0x3, LinearGradientMode.WEIGHT)
        self.assertEqual((backward >> 28) & 1, 1)
        self.assertEqual((backward >> 29) & 1, 1)
        with self.assertRaises(ValueError):
            linear_backward_command(0, 1, 1)
        with self.assertRaises(ValueError):
            linear_backward_command(1, 9, 1)

    def test_tensor_stream_layouts(self):
        activations = pack_activation_beats([[0, 1, 2, 3, 4, 5, 6, 255]])
        self.assertEqual(activations[0].data, 0xFF06050403020100)
        self.assertTrue(activations[0].last)
        outer = pack_outer_weight_beats([list(range(16))])
        self.assertEqual([beat.data for beat in outer], [0x0706050403020100, 0x0F0E0D0C0B0A0908])
        self.assertEqual([beat.last for beat in outer], [False, True])
        matrix = [[16 * row + column for column in range(16)] for row in range(8)]
        dot8 = pack_dot8_weight_beats([matrix])
        self.assertEqual(len(dot8), 16)
        self.assertEqual(dot8[0].data, 0x0706050403020100)
        self.assertEqual(dot8[1].data, 0x0F0E0D0C0B0A0908)
        self.assertEqual(dot8[2].data, 0x1716151413121110)
        self.assertTrue(dot8[-1].last)

    def test_descriptor_program_layout(self):
        records = (
            ProgramRecord(
                clear_status_command(),
                (StreamBeat(0xAA, False), StreamBeat(0xBB, True)),
                0,
            ),
            ProgramRecord(clear_status_command(), (), 3),
        )
        beats, outputs = pack_descriptor_program(records)
        self.assertEqual(outputs, 3)
        self.assertEqual(len(beats), 8)
        self.assertEqual(beats[0].data, 2)
        self.assertEqual(beats[5].data, (1 << 63) | (3 << 32))
        self.assertEqual([beat.last for beat in beats], [False] * 7 + [True])

    def test_operator_stream_layouts(self):
        query = [index & 1 for index in range(256)]
        key = [1 - bit for bit in query]
        alpha = pack_alpha_beats(query, key)
        self.assertEqual(len(alpha), 8)
        self.assertEqual(alpha[0].data, 0xAAAAAAAAAAAAAAAA)
        self.assertEqual(alpha[4].data, 0x5555555555555555)
        self.assertTrue(alpha[-1].last)

        vector = pack_vector_beats(range(-4, 4), range(10, 18), clamp_minimum=-7, clamp_maximum=8)
        self.assertEqual(len(vector), 9)
        self.assertEqual(vector[0].data, 0x00000008FFFFFFF9)
        self.assertEqual(vector[1].data, 0xFFFFFFFDFFFFFFFC)
        self.assertTrue(vector[-1].last)

        layernorm = pack_layernorm_beats([-1], [-2], [-3])
        self.assertEqual(layernorm[0].data, 0xFFFDFFFFFEFFFFFF)
        self.assertEqual(layernorm[1].data, 0xFF)
        self.assertEqual([beat.last for beat in layernorm], [False, True])
        self.assertEqual(pack_softmax_beats([-1, 2])[0].data, 0xFFFFFF)

        lif = pack_lif_beats(range(-16, 16))
        self.assertEqual(len(lif), 8)
        self.assertEqual(lif[0].data, 0xFFF3FFF2FFF1FFF0)
        optimizer = pack_optimizer_beats(range(-4, 4), range(-8, 0), range(8))
        self.assertEqual(len(optimizer), 8)
        self.assertEqual(optimizer[0].data, 0x000000FFFFF8FFFC)
        self.assertTrue(optimizer[-1].last)

        load = pack_lut_load_beats([(7, -2), (8, 3)])
        self.assertEqual(load[0].data, 0x00FFFE07)
        self.assertEqual([beat.last for beat in load], [False, True])
        self.assertEqual(pack_lut_eval_beat(range(8))[0].data, 0x0706050403020100)

        backward = pack_linear_backward_beats(
            [[-128, -1, 0, 1], [127, 2, -3, 4]],
            [[-32768, -1, 32767], [1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[-32768, -1, 32767], [10, 11, 12]],
        )
        self.assertEqual(len(backward), 16)
        self.assertEqual(backward[0].data, 0x00010000FFFFFF80)
        self.assertEqual(backward[1].data, 0)
        self.assertEqual(backward[2].data, 0x0004FFFD0002007F)
        self.assertEqual(backward[4].data, 0x00007FFFFFFF8000)
        self.assertEqual(backward[5].data, 0)
        self.assertEqual([beat.last for beat in backward], [False] * 15 + [True])
        input_only = pack_linear_backward_beats(
            [[-128, -1, 0, 1], [127, 2, -3, 4]],
            [[-32768, -1, 32767], [1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[-32768, -1, 32767], [10, 11, 12]],
            mode=LinearGradientMode.INPUT,
        )
        weight_only = pack_linear_backward_beats(
            [[-128, -1, 0, 1], [127, 2, -3, 4]],
            [[-32768, -1, 32767], [1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[-32768, -1, 32767], [10, 11, 12]],
            mode=LinearGradientMode.WEIGHT,
        )
        self.assertEqual(len(input_only), 12)
        self.assertEqual(input_only[0].data, backward[4].data)
        self.assertEqual(len(weight_only), 8)
        self.assertEqual(weight_only[0].data, backward[0].data)

    def test_result_decoders(self):
        tensor_word = ((-7) & ((1 << 48) - 1)) | (5 << 48) | (12 << 52) | (1 << 58)
        self.assertEqual(unpack_tensor_result(tensor_word), (-7, 5, 12, True))
        self.assertEqual(unpack_alpha_result(11 | (12 << 18) | (13 << 27)), (11, 12, 13))
        self.assertEqual(unpack_vector_result([0xFFFFFFFF00000001, 0, 0, 0])[:2], (1, -1))
        self.assertEqual(unpack_layernorm_result(0xFFFFFFFF | (9 << 32) | (1 << 44)), (-1, 9, True))
        self.assertEqual(unpack_softmax_result(123 | (7 << 16) | (1 << 25)), (123, 7, True))
        self.assertEqual(unpack_lif_result(0x12345678ABCDEF01), (0xABCDEF01, 0x12345678))
        expected_membranes = tuple(
            (-1 if lane % 3 == 0 else (lane * 7919) % (1 << 23))
            for lane in range(32)
        )
        packed_membranes = sum(
            (value & 0xFFFFFF) << (24 * lane)
            for lane, value in enumerate(expected_membranes)
        )
        trace_words = [0x12345678ABCDEF01]
        trace_words.extend(
            (packed_membranes >> (64 * index)) & 0xFFFFFFFFFFFFFFFF
            for index in range(12)
        )
        self.assertEqual(
            unpack_lif_trace(trace_words),
            (0xABCDEF01, 0x12345678, expected_membranes),
        )
        with self.assertRaises(ValueError):
            unpack_lif_trace(trace_words[:-1])
        weights, velocities = unpack_optimizer_result([0xFFFFFF000001FFFF] * 8)
        self.assertEqual(weights[0], -1)
        self.assertEqual(velocities[0], 1)
        self.assertEqual(unpack_lut_result([0x0003FFFE0001FFFF, 0]), (-1, 1, -2, 3, 0, 0, 0, 0))

        expected_dx = ((-1, 2, -(1 << 31), (1 << 31) - 1), (9, -10, 11, -12))
        expected_dw = (
            (-1, 2, -3),
            ((1 << 23) - 1, -(1 << 23), 7),
            (8, 9, 10),
            (-11, -12, -13),
        )
        backward_words = []
        for row in expected_dx:
            padded = row + (0,) * (8 - len(row))
            backward_words.extend(
                (padded[lane] & 0xFFFFFFFF)
                | ((padded[lane + 1] & 0xFFFFFFFF) << 32)
                for lane in range(0, 8, 2)
            )
        for row in expected_dw:
            padded = row + (0,) * (8 - len(row))
            packed = sum((value & 0xFFFFFF) << (24 * lane)
                         for lane, value in enumerate(padded))
            backward_words.extend(
                (packed >> (64 * word)) & 0xFFFFFFFFFFFFFFFF for word in range(3)
            )
        self.assertEqual(
            unpack_linear_backward_result(backward_words, 2, 4, 3),
            (expected_dx, expected_dw),
        )
        self.assertEqual(
            unpack_linear_backward_result(
                backward_words[:8], 2, 4, 3, mode=LinearGradientMode.INPUT
            ),
            (expected_dx, ()),
        )
        self.assertEqual(
            unpack_linear_backward_result(
                backward_words[8:], 2, 4, 3, mode=LinearGradientMode.WEIGHT
            ),
            ((), expected_dw),
        )
        with self.assertRaises(ValueError):
            unpack_linear_backward_result(backward_words[:-1], 2, 4, 3)


if __name__ == "__main__":
    unittest.main()
