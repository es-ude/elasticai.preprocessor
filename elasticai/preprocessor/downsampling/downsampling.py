from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SettingsDownSampling:
    """Settings class for configuring the properties of the downsampling module
    Attributes:
        sampling_rate:  Floating value with input sampling rate of the transient data stream
        dsr:            Integer with downsampling ratio for reducing the input sampling rate (SR_out = SR_in / OSR)
    """

    sampling_rate: float
    dsr: int


DefaultSettingsDownSampling = SettingsDownSampling(
    sampling_rate=1000.0,
    dsr=10,
)


class DownSampling:
    def __init__(self, settings: SettingsDownSampling):
        self._settings = settings

    @property
    def sampling_rate_out(self) -> float:
        return self._settings.sampling_rate / self._settings.dsr

    def do_subsampling(self, data: np.ndarray, augment: bool = False) -> np.ndarray:
        """Downsample datasets by taking every dsr-th value along the last axis.

        When augment is True, additional samples are generated from the
        remaining offsets and concatenated along the sample axis. Missing tail
        values are zero-padded so all generated samples have equal length.
        """
        factor = self._settings.dsr
        if factor < 1:
            raise ValueError("dsr must be >= 1")
        if factor == 1:
            return data
        if data.ndim < 2:
            raise ValueError("subsampling expects a sample axis")

        output_length = data[..., 0::factor].shape[-1]
        downsampled_offsets = [
            self._pad_last_axis(data[..., offset::factor], output_length) for offset in range(factor)
        ]
        if not augment:
            return downsampled_offsets[0]
        return np.concatenate(downsampled_offsets, axis=0)

    @staticmethod
    def _pad_last_axis(data: np.ndarray, output_length: int) -> np.ndarray:
        pad_length = output_length - data.shape[-1]
        if pad_length <= 0:
            return data
        padding = np.zeros(data.shape[:-1] + (pad_length,), dtype=data.dtype)
        return np.concatenate([data, padding], axis=-1)

    def create_design(
        self,
        target: str,
        bitwidth: int,
        id: str,
        path2save: Path,
        signed: bool = True,
    ) -> None:
        """Generate a C design for subsampling."""
        supported_targets = ["mcu", "pc", "fpga"]
        target = target.lower()
        if target not in supported_targets:
            raise ValueError(f"Target {target} is not supported: only {supported_targets}")
        if target == "fpga":
            raise NotImplementedError("FPGA downsampling generation is not implemented")
        self._create_design_c(id=id, bitwidth=bitwidth, signed=signed, path2save=path2save)

    def _create_design_c(self, id: str, bitwidth: int, signed: bool, path2save: Path) -> None:
        from elasticai.creator_plugins.downsampling.src import c_compile

        c_compile.build_downsampling_subsampling(
            downsampling_ratio=self._settings.dsr,
            bitwidth=bitwidth,
            signed=signed,
            downsampling_id=id,
            path2save=path2save,
            define_path=".",
        )

    def do_simple(self, uin: np.ndarray) -> np.ndarray:
        """Performing a simple downsampling of the adc data stream
        param uin:          Numpy array with transient signal input (high sampling rate)
        return:             Numpy array with transient signal output (low sampling rate)
        """
        n = uin.size // self._settings.dsr * self._settings.dsr
        data = uin[:n]
        return data.reshape(-1, self._settings.dsr).mean(axis=1)

    def do_cic(self, uin: np.ndarray, num_stages: int = 5) -> np.ndarray:
        """Performing the CIC filter at the output of oversampled ADC
        param uin:          Numpy array with transient signal input (high sampling rate)
        param num_stages:   Number of stages to perform the CIC downsampling
        return:             Numpy array with transient signal output (low sampling rate)
        """
        output_transient = list()
        gain = (self._settings.dsr * 1) ** num_stages

        class integrator:
            def __init__(self):
                self.yn = 0
                self.ynm = 0

            def update(self, inp):
                self.ynm = self.yn
                self.yn = self.ynm + inp
                return self.yn

        class comb:
            def __init__(self):
                self.xn = 0
                self.xnm = 0

            def update(self, inp):
                self.xnm = self.xn
                self.xn = inp
                return self.xn - self.xnm

        intes = [integrator() for a in range(num_stages)]
        combs = [comb() for a in range(num_stages)]
        for s, v in enumerate(uin):
            z = v
            for i in range(num_stages):
                z = intes[i].update(z)

            if (s % self._settings.dsr) == 0:
                for c in range(num_stages):
                    z = combs[c].update(z)
                    j = z
                output_transient.append(j / gain)
        return np.array(output_transient)

    @staticmethod
    def do_decimation_polyphase_order_one(uin: np.ndarray) -> np.ndarray:
        """Performing first order Non-Recursive Polyphase Decimation on input
        param uin:          Numpy array with transient signal input (high sampling rate)
        return:             Numpy array with transient signal output (low sampling rate)
        """
        last_sample_hs = 0.
        uout = list()
        for idx, val in enumerate(uin):
            if idx % 2 == 1:
                uout.append(val + last_sample_hs)
            last_sample_hs = val
        return np.array(uout)

    @staticmethod
    def do_decimation_polyphase_order_two(uin: np.ndarray) -> np.ndarray:
        """Performing second order Non-Recursive Polyphase Decimation on input
        param uin:          Numpy array with transient signal input (high sampling rate)
        return:             Numpy array with transient signal output (low sampling rate)
        """
        last_even_prev = 0.
        last_even = 0.
        uout = list()
        for idx, val in enumerate(uin):
            if idx % 2 == 0:
                last_even_prev = last_even
                last_even = val
            else:
                uout.append(val + 2 * last_even + last_even_prev)
        return np.array(uout)

    def do_decimation_polyphase(self, uin: np.ndarray, take_first_order: bool = False) -> np.ndarray:
        """Performing Non-Recursive Polyphase Decimation on input (depends on DSR)
        param uin:          Numpy array with transient signal input (high sampling rate)
        return:             Numpy array with transient signal output (low sampling rate)
        """
        val = np.log2(self._settings.dsr)
        if not val.is_integer():
            raise ValueError("self._settings.dsr should be 2^x")

        x = uin
        for _ in range(int(val)):
            if take_first_order:
                x = self._do_decimation_polyphase_order_one(x)
            else:
                x = self._do_decimation_polyphase_order_two(x)
        return x
