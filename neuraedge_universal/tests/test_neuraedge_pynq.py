import unittest

import numpy as np

from neuraedge_protocol import ProgramRecord, StreamBeat, clear_status_command
from neuraedge_pynq import NeuraEdgeDriver, NeuraEdgeHardwareError


class FakeMmio:
    def __init__(self, events, *, error=False):
        self.events = events
        self.error = error
        self.registers = {0x30: 0x00010007, 0x34: 0x4E455552,
                          0x28: 200000000, 0x2C: 0x1FF, 0x04: 0}

    def read(self, address):
        return self.registers.get(address, 0)

    def write(self, address, value):
        self.events.append(("mmio", address, value))
        self.registers[address] = value
        if address == 0 and value & 0x5:
            self.registers[4] = 0x2 if self.error else 0x4


class FakeBuffer(np.ndarray):
    def __new__(cls, shape, dtype):
        return np.zeros(shape, dtype=dtype).view(cls)

    def flush(self):
        pass

    def invalidate(self):
        pass

    def freebuffer(self):
        self.freed = True


class FakeChannel:
    def __init__(self, events, name, receive_words=()):
        self.events = events
        self.name = name
        self.receive_words = receive_words
        self.idle = True

    def transfer(self, buffer, nbytes):
        self.events.append((self.name, "transfer", nbytes))
        if self.name == "recv":
            buffer[:len(self.receive_words)] = self.receive_words

    def wait(self):
        self.events.append((self.name, "wait"))


class FakeDma:
    def __init__(self, events, receive_words=()):
        self.sendchannel = FakeChannel(events, "send")
        self.recvchannel = FakeChannel(events, "recv", receive_words)


class PynqDriverTests(unittest.TestCase):
    def make_driver(self, *, receive_words=(), error=False):
        events = []
        mmio = FakeMmio(events, error=error)
        dma = FakeDma(events, receive_words)
        driver = NeuraEdgeDriver(mmio, dma, lambda *, shape, dtype: FakeBuffer(shape, dtype))
        return driver, events

    def test_identity_and_metadata(self):
        driver, _ = self.make_driver()
        self.assertEqual(driver.target_clock_hz, 200000000)
        self.assertEqual(driver.capabilities, 0x1FF)

    def test_performance_snapshot_decodes_hardware_registers(self):
        driver, _ = self.make_driver()
        driver.mmio.registers.update({
            0x38: (29 << 16) | 31,
            0x3C: 17,
            0x40: 1234,
            0x44: 14,
            0x48: 0x89ABCDEF,
            0x4C: 0x01234567,
            0x50: 0x76543210,
            0x54: 0xFEDCBA98,
            0x58: 222,
            0x5C: 0xFFFFFFF9,
            0x60: 81,
            0x64: 9,
            0x68: 333,
            0x6C: 0xFFFFFFF5,
            0x70: 0xABCDEF,
            0x74: 0x11111111,
            0x78: 0x22222222,
            0x7C: 44,
            0x80: 256,
            0x84: 0x12345678,
            0x88: 0x87654321,
            0x94: 19,
            0x98: 9876,
        })
        snapshot = driver.performance_snapshot()
        self.assertEqual(snapshot.tensor_loaded_activations, 31)
        self.assertEqual(snapshot.tensor_loaded_weights, 29)
        self.assertEqual(snapshot.tensor_active_issues, 17)
        self.assertEqual(snapshot.dense_macs, 0x0123456789ABCDEF)
        self.assertEqual(snapshot.spike_adds, 0xFEDCBA9876543210)
        self.assertEqual(snapshot.layernorm_mean, -7)
        self.assertEqual(snapshot.softmax_maximum, -11)
        self.assertEqual(snapshot.lif_total_spikes, 0x2222222211111111)
        self.assertEqual(snapshot.lut_loaded_entries, 256)
        self.assertEqual(snapshot.training_tiles, 19)
        self.assertEqual(snapshot.training_cycles, 9876)

    def test_receive_is_armed_before_submit_and_send(self):
        driver, events = self.make_driver(receive_words=(0x1234, 0x5678))
        results = driver.execute(
            clear_status_command(),
            [StreamBeat(1, False), StreamBeat(2, True)],
            output_beats=2,
        )
        self.assertEqual(results, (0x1234, 0x5678))
        receive_index = events.index(("recv", "transfer", 16))
        submit_index = next(index for index, event in enumerate(events)
                            if event[0] == "mmio" and event[1] == 0 and event[2] & 1)
        send_index = events.index(("send", "transfer", 16))
        self.assertLess(receive_index, submit_index)
        self.assertLess(submit_index, send_index)

    def test_rejects_bad_last_and_hardware_error(self):
        driver, _ = self.make_driver()
        with self.assertRaises(ValueError):
            driver.execute(clear_status_command(), [StreamBeat(1, False)])
        driver, _ = self.make_driver(error=True)
        with self.assertRaises(NeuraEdgeHardwareError):
            driver.execute(clear_status_command())

    def test_descriptor_program_uses_one_dma_pair(self):
        driver, events = self.make_driver(receive_words=(0xAA, 0xBB))
        driver.mmio.registers[0x8C] = 2
        records = (
            ProgramRecord(
                clear_status_command(),
                (StreamBeat(0x11, False), StreamBeat(0x22, True)),
                0,
            ),
            ProgramRecord(clear_status_command(), (), 2),
        )
        result = driver.execute_program(records)
        self.assertEqual(result, (0xAA, 0xBB))
        self.assertIn(("recv", "transfer", 16), events)
        self.assertIn(("send", "transfer", 64), events)
        self.assertIn(("mmio", 0x90, 2), events)
        self.assertTrue(any(event[0] == "mmio" and event[1] == 0
                            and event[2] & 4 for event in events))

    def test_descriptor_program_partitions_at_dma_packet_boundary(self):
        driver, events = self.make_driver(receive_words=(0xAA,))
        driver.MAX_DMA_TRANSFER_BYTES = 32
        driver.mmio.registers[0x8C] = 1
        records = (
            ProgramRecord(clear_status_command(), (), 1),
            ProgramRecord(clear_status_command(), (), 1),
        )
        result = driver.execute_program(records)
        self.assertEqual(result, (0xAA, 0xAA))
        self.assertEqual(events.count(("send", "transfer", 24)), 2)
        self.assertEqual(events.count(("recv", "transfer", 8)), 2)


if __name__ == "__main__":
    unittest.main()
