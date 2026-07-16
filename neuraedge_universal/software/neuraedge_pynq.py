"""PYNQ DMA/MMIO driver for the complete NeuraEdge overlay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Callable, Sequence

import numpy as np

from neuraedge_protocol import (
    Command128,
    PackedStreamBeats,
    ProgramRecord,
    StreamBeat,
)


@dataclass(frozen=True)
class RegisterMap:
    control: int = 0x00
    status: int = 0x04
    active_opcode: int = 0x08
    command_count: int = 0x0C
    command0: int = 0x10
    input_beats: int = 0x20
    output_beats: int = 0x24
    target_clock: int = 0x28
    capabilities: int = 0x2C
    version: int = 0x30
    identity: int = 0x34
    tensor_loaded: int = 0x38
    tensor_active: int = 0x3C
    tensor_cycles: int = 0x40
    tensor_skipped: int = 0x44
    dense_macs_low: int = 0x48
    dense_macs_high: int = 0x4C
    spike_adds_low: int = 0x50
    spike_adds_high: int = 0x54
    layernorm_cycles: int = 0x58
    layernorm_mean: int = 0x5C
    layernorm_variance: int = 0x60
    layernorm_sigma: int = 0x64
    softmax_cycles: int = 0x68
    softmax_maximum: int = 0x6C
    softmax_denominator: int = 0x70
    lif_spikes_low: int = 0x74
    lif_spikes_high: int = 0x78
    optimizer_updates: int = 0x7C
    lut_entries: int = 0x80
    layernorm_reciprocal: int = 0x84
    softmax_reciprocal: int = 0x88
    program_records: int = 0x8C
    program_output_beats: int = 0x90
    training_tiles: int = 0x94
    training_cycles: int = 0x98


@dataclass(frozen=True)
class PerformanceSnapshot:
    tensor_loaded_activations: int
    tensor_loaded_weights: int
    tensor_active_issues: int
    tensor_compute_cycles: int
    tensor_skipped_issues: int
    dense_macs: int
    spike_adds: int
    layernorm_cycles: int
    layernorm_mean: int
    layernorm_variance: int
    layernorm_sigma: int
    layernorm_reciprocal_q31: int
    softmax_cycles: int
    softmax_maximum: int
    softmax_denominator: int
    softmax_reciprocal_q31: int
    lif_total_spikes: int
    optimizer_updates: int
    lut_loaded_entries: int
    training_tiles: int
    training_cycles: int


class NeuraEdgeHardwareError(RuntimeError):
    """Raised when the RTL reports a sticky protocol or configuration error."""


class NeuraEdgeTimeout(TimeoutError):
    """Raised when a DMA channel or accelerator command exceeds its deadline."""


class NeuraEdgeDriver:
    STATUS_BUSY = 1 << 0
    STATUS_ERROR = 1 << 1
    STATUS_DONE = 1 << 2
    STATUS_PENDING = 1 << 3
    STATUS_PROGRAM_ACTIVE = 1 << 5
    EXPECTED_ID = 0x4E455552
    EXPECTED_ABI_MAJOR = 1
    MAX_DMA_TRANSFER_BYTES = (1 << 26) - 1
    PACK_COPY_WORDS = 1 << 16

    def __init__(self, mmio, dma, allocator: Callable[..., object], *, interrupt_enable: bool = True):
        self.mmio = mmio
        self.dma = dma
        self.allocator = allocator
        self.registers = RegisterMap()
        self.interrupt_enable = bool(interrupt_enable)
        self._check_identity()

    @classmethod
    def from_bitfile(
        cls,
        bitfile: str | Path,
        *,
        download: bool = True,
        accelerator_name: str = "accelerator",
        dma_name: str = "dma",
    ) -> "NeuraEdgeDriver":
        try:
            from pynq import Overlay, allocate
        except ImportError as error:
            raise RuntimeError("the pynq package is required on the PYNQ-Z2 board") from error

        overlay = Overlay(str(Path(bitfile)), download=download)
        accelerator = getattr(overlay, accelerator_name)
        dma = getattr(overlay, dma_name)
        driver = cls(accelerator.mmio, dma, allocate)
        driver.overlay = overlay
        return driver

    def _check_identity(self) -> None:
        identity = int(self.mmio.read(self.registers.identity))
        version = int(self.mmio.read(self.registers.version))
        if identity != self.EXPECTED_ID:
            raise NeuraEdgeHardwareError(
                f"unexpected accelerator identity 0x{identity:08x}; expected 0x{self.EXPECTED_ID:08x}"
            )
        if version >> 16 != self.EXPECTED_ABI_MAJOR:
            raise NeuraEdgeHardwareError(
                f"unsupported accelerator ABI 0x{version:08x}; expected major {self.EXPECTED_ABI_MAJOR}"
            )

    @property
    def target_clock_hz(self) -> int:
        return int(self.mmio.read(self.registers.target_clock))

    @property
    def capabilities(self) -> int:
        return int(self.mmio.read(self.registers.capabilities))

    def status(self) -> int:
        return int(self.mmio.read(self.registers.status))

    @staticmethod
    def _signed32(value: int) -> int:
        masked = int(value) & 0xFFFFFFFF
        return masked - (1 << 32) if masked & (1 << 31) else masked

    def performance_snapshot(self) -> PerformanceSnapshot:
        """Read one coherent-enough diagnostic snapshot while the core is idle."""
        if self.status() & (self.STATUS_BUSY | self.STATUS_PENDING):
            raise NeuraEdgeHardwareError("performance counters must be sampled while the accelerator is idle")
        registers = self.registers
        loaded = int(self.mmio.read(registers.tensor_loaded))
        dense_macs = int(self.mmio.read(registers.dense_macs_low))
        dense_macs |= int(self.mmio.read(registers.dense_macs_high)) << 32
        spike_adds = int(self.mmio.read(registers.spike_adds_low))
        spike_adds |= int(self.mmio.read(registers.spike_adds_high)) << 32
        lif_spikes = int(self.mmio.read(registers.lif_spikes_low))
        lif_spikes |= int(self.mmio.read(registers.lif_spikes_high)) << 32
        return PerformanceSnapshot(
            loaded & 0xFFF,
            (loaded >> 16) & 0xFFF,
            int(self.mmio.read(registers.tensor_active)) & 0xFFF,
            int(self.mmio.read(registers.tensor_cycles)),
            int(self.mmio.read(registers.tensor_skipped)),
            dense_macs,
            spike_adds,
            int(self.mmio.read(registers.layernorm_cycles)),
            self._signed32(self.mmio.read(registers.layernorm_mean)),
            int(self.mmio.read(registers.layernorm_variance)),
            int(self.mmio.read(registers.layernorm_sigma)),
            int(self.mmio.read(registers.layernorm_reciprocal)),
            int(self.mmio.read(registers.softmax_cycles)),
            self._signed32(self.mmio.read(registers.softmax_maximum)),
            int(self.mmio.read(registers.softmax_denominator)) & 0xFFFFFF,
            int(self.mmio.read(registers.softmax_reciprocal)),
            lif_spikes,
            int(self.mmio.read(registers.optimizer_updates)),
            int(self.mmio.read(registers.lut_entries)) & 0x1FF,
            int(self.mmio.read(registers.training_tiles)),
            int(self.mmio.read(registers.training_cycles)),
        )

    def _control_value(self, transient_bits: int = 0) -> int:
        return ((1 << 8) if self.interrupt_enable else 0) | transient_bits

    def clear_wrapper_status(self) -> None:
        self.mmio.write(self.registers.control, self._control_value(1 << 1))

    @staticmethod
    def _validate_input_beats(beats: Sequence[StreamBeat]) -> None:
        if isinstance(beats, PackedStreamBeats):
            return
        for index, beat in enumerate(beats):
            if beat.last != (index == len(beats) - 1):
                raise ValueError("input StreamBeat.last must be true only on the final beat")

    @staticmethod
    def _release_buffer(buffer) -> None:
        if buffer is None:
            return
        if hasattr(buffer, "freebuffer"):
            buffer.freebuffer()
        elif hasattr(buffer, "close"):
            buffer.close()

    @staticmethod
    def _wait_dma_channel(channel, deadline: float, name: str) -> None:
        while not bool(channel.idle):
            if monotonic() >= deadline:
                if hasattr(channel, "stop"):
                    channel.stop()
                raise NeuraEdgeTimeout(f"{name} DMA channel timed out")
            sleep(0.0005)
        channel.wait()

    def _wait_idle(self, deadline: float) -> None:
        while self.status() & (self.STATUS_BUSY | self.STATUS_PENDING):
            if monotonic() >= deadline:
                raise NeuraEdgeTimeout("accelerator did not become idle")
            sleep(0.0005)

    def execute(
        self,
        command: Command128,
        input_beats: Sequence[StreamBeat] = (),
        *,
        output_beats: int = 0,
        timeout_seconds: float = 5.0,
    ) -> tuple[int, ...]:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if output_beats < 0:
            raise ValueError("output_beats must be non-negative")
        self._validate_input_beats(input_beats)
        deadline = monotonic() + timeout_seconds
        self._wait_idle(deadline)

        input_buffer = None
        output_buffer = None
        try:
            if input_beats:
                input_buffer = self.allocator(shape=(len(input_beats),), dtype=np.uint64)
                if isinstance(input_beats, PackedStreamBeats):
                    input_buffer[:] = np.frombuffer(input_beats.words, dtype=np.uint64)
                else:
                    input_buffer[:] = np.fromiter(
                        (beat.data for beat in input_beats),
                        dtype=np.uint64,
                        count=len(input_beats),
                    )
                if hasattr(input_buffer, "flush"):
                    input_buffer.flush()
            if output_beats:
                output_buffer = self.allocator(shape=(output_beats,), dtype=np.uint64)
                self.dma.recvchannel.transfer(output_buffer, nbytes=output_beats * 8)

            self.clear_wrapper_status()
            for index, word in enumerate(command.words_le):
                self.mmio.write(self.registers.command0 + 4 * index, word)
            self.mmio.write(self.registers.control, self._control_value(1))

            if input_buffer is not None:
                self.dma.sendchannel.transfer(input_buffer, nbytes=len(input_beats) * 8)
                self._wait_dma_channel(self.dma.sendchannel, deadline, "MM2S")
            if output_buffer is not None:
                self._wait_dma_channel(self.dma.recvchannel, deadline, "S2MM")

            while True:
                status = self.status()
                if status & self.STATUS_ERROR:
                    raise NeuraEdgeHardwareError(
                        f"accelerator rejected opcode {int(command.opcode):x}; status=0x{status:08x}"
                    )
                if status & self.STATUS_DONE:
                    break
                if monotonic() >= deadline:
                    raise NeuraEdgeTimeout(f"opcode {int(command.opcode):x} did not complete")
                sleep(0.0005)

            if output_buffer is None:
                return ()
            if hasattr(output_buffer, "invalidate"):
                output_buffer.invalidate()
            return tuple(int(value) for value in output_buffer[:output_beats])
        finally:
            self._release_buffer(input_buffer)
            self._release_buffer(output_buffer)

    def execute_program(
        self,
        records: Sequence[ProgramRecord],
        *,
        timeout_seconds: float = 30.0,
    ) -> tuple[int, ...]:
        """Execute records in the minimum number of legal descriptor packets."""

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not records:
            raise ValueError("a descriptor program must contain at least one record")
        deadline = monotonic() + timeout_seconds
        output: list[int] = []
        for packet in self._partition_program(records):
            output.extend(self._execute_program_packet(packet, deadline))
        return tuple(output)

    def _partition_program(
        self,
        records: Sequence[ProgramRecord],
    ) -> tuple[tuple[ProgramRecord, ...], ...]:
        packets: list[tuple[ProgramRecord, ...]] = []
        current: list[ProgramRecord] = []
        current_input_bytes = 0
        current_output_bytes = 0
        for record in records:
            record_input_bytes = (3 + len(record.input_beats)) * 8
            record_output_bytes = record.output_beats * 8
            if (record_input_bytes > self.MAX_DMA_TRANSFER_BYTES
                    or record_output_bytes > self.MAX_DMA_TRANSFER_BYTES):
                raise ValueError("one descriptor record exceeds the configured AXI DMA length")
            if current and (
                    current_input_bytes + record_input_bytes > self.MAX_DMA_TRANSFER_BYTES
                    or current_output_bytes + record_output_bytes > self.MAX_DMA_TRANSFER_BYTES):
                packets.append(tuple(current))
                current = []
                current_input_bytes = 0
                current_output_bytes = 0
            current.append(record)
            current_input_bytes += record_input_bytes
            current_output_bytes += record_output_bytes
        packets.append(tuple(current))
        return tuple(packets)

    @staticmethod
    def _program_sizes(records: Sequence[ProgramRecord]) -> tuple[int, int]:
        input_words = sum(3 + len(record.input_beats) for record in records)
        output_words = sum(record.output_beats for record in records)
        return input_words, output_words

    def _fill_program_buffer(self, buffer, records: Sequence[ProgramRecord]) -> None:
        cursor = 0
        for record_index, record in enumerate(records):
            header = len(record.input_beats)
            header |= record.output_beats << 32
            header |= int(record_index == len(records) - 1) << 63
            buffer[cursor] = header
            buffer[cursor + 1] = record.command.value & ((1 << 64) - 1)
            buffer[cursor + 2] = record.command.value >> 64
            cursor += 3
            if isinstance(record.input_beats, PackedStreamBeats):
                words = np.frombuffer(record.input_beats.words, dtype=np.uint64)
                buffer[cursor:cursor + len(words)] = words
                cursor += len(words)
                continue
            for start in range(0, len(record.input_beats), self.PACK_COPY_WORDS):
                beats = record.input_beats[start:start + self.PACK_COPY_WORDS]
                words = np.fromiter(
                    (beat.data for beat in beats), dtype=np.uint64, count=len(beats))
                buffer[cursor:cursor + len(beats)] = words
                cursor += len(beats)
        if cursor != len(buffer):
            raise RuntimeError(f"packed {cursor} descriptor words into a {len(buffer)}-word buffer")

    def _execute_program_packet(
        self,
        records: Sequence[ProgramRecord],
        deadline: float,
    ) -> tuple[int, ...]:
        input_words, output_beats = self._program_sizes(records)
        self._wait_idle(deadline)
        input_buffer = None
        output_buffer = None
        try:
            input_buffer = self.allocator(shape=(input_words,), dtype=np.uint64)
            self._fill_program_buffer(input_buffer, records)
            if hasattr(input_buffer, "flush"):
                input_buffer.flush()
            if output_beats:
                output_buffer = self.allocator(shape=(output_beats,), dtype=np.uint64)
                self.dma.recvchannel.transfer(output_buffer, nbytes=output_beats * 8)

            self.clear_wrapper_status()
            self.mmio.write(self.registers.program_output_beats, output_beats)
            self.mmio.write(self.registers.control, self._control_value(1 << 2))
            self.dma.sendchannel.transfer(input_buffer, nbytes=input_words * 8)
            self._wait_dma_channel(self.dma.sendchannel, deadline, "program MM2S")
            if output_buffer is not None:
                self._wait_dma_channel(self.dma.recvchannel, deadline, "program S2MM")

            while True:
                status = self.status()
                if status & self.STATUS_ERROR:
                    raise NeuraEdgeHardwareError(
                        f"descriptor program failed; status=0x{status:08x}")
                if status & self.STATUS_DONE:
                    break
                if monotonic() >= deadline:
                    raise NeuraEdgeTimeout("descriptor program did not complete")
                sleep(0.0005)
            completed = int(self.mmio.read(self.registers.program_records))
            if completed != len(records):
                raise NeuraEdgeHardwareError(
                    f"descriptor program completed {completed} records; expected {len(records)}")
            if output_buffer is None:
                return ()
            if hasattr(output_buffer, "invalidate"):
                output_buffer.invalidate()
            return tuple(int(value) for value in output_buffer[:output_beats])
        finally:
            self._release_buffer(input_buffer)
            self._release_buffer(output_buffer)
