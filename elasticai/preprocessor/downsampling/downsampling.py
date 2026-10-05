from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

import elasticai.creator_plugins.datarate as datarate_filters
from elasticai.creator_plugins.datarate.src import c_compile
from elasticai.preprocessor.sequential import PreprocessingModule, SequentialSignal


class DownSamplingTargets(Enum):
    Subsampling = "subsampling"
    Simple = "simple"
    CIC = "cic"
    Polyphase = "polyphase"


@dataclass
class DownSamplingSettings:
    """Settings class for configuring the properties of the downsampling module
    Attributes:
        method:         Used method for downsampling (TargetsDownSampling)
        sampling_rate:  Floating value with input sampling rate of the transient data stream
        num_stages:     Number of stages
        dsr:            Integer with downsampling ratio for reducing the input sampling rate (SR_out = SR_in / OSR)
    """

    method: DownSamplingTargets
    sampling_rate: float
    num_stages: int
    dsr: int


DefaultDownSamplingSettings = DownSamplingSettings(
    method=DownSamplingTargets.Simple,
    sampling_rate=1000.0,
    num_stages=5,
    dsr=10,
)


class DownSampling(PreprocessingModule):
    def __init__(self, settings: DownSamplingSettings):
        super().__init__()
        self._settings = settings
        if isinstance(settings.method, str):
            self._settings.method = DownSamplingTargets(settings.method)

    def __call__(self, x: SequentialSignal) -> SequentialSignal:
        return SequentialSignal(
            data=self.decimate(x.data),
            sample_rate=self.sampling_rate_out,
        )

    @property
    def sampling_rate_out(self) -> float:
        return self._settings.sampling_rate / self._settings.dsr

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
        """Generate the hardware design to downsampling on hardware
        :param target:           Target platform ["mcu", "pc", "fpga", "asic"]
        :param bitwidth:         Bitwidth
        :param id:               ID of the target structure
        :param path2save:        Path to save downsampling subsampling
        :param signed:           Signal to use for downsampling
        :return:                 None
        """

        supported_targets = ["mcu", "pc", "fpga", "asic"]
        if target.lower() not in supported_targets:
            raise ValueError(f"Target {target} is not supported: only {supported_targets}")
        if self._settings.dsr < 1:
            raise ValueError("dsr must be >= 1")
        if self._settings.num_stages < 1:
            raise ValueError("num_stages must be >= 1")
        assert bitwidth in range(2, 33), "Bitwidth must be between 2 and 32"

        if target.lower() in ["mcu", "pc"]:
            self._create_design_c(
                id=id,
                bitwidth=bitwidth,
                signed=signed,
                path2save=path2save,
            )
        elif target.lower() in ["fpga"]:
            self._create_design_fpga_verilog(
                id=id,
                bitwidth=bitwidth,
                path2save=path2save,
            )
        elif target.lower() in ["asic"]:
            self._create_design_asic_verilog(
                id=id,
                bitwidth=bitwidth,
                path2save=path2save,
            )

    def _create_design_c(
        self,
        id: str,
        bitwidth: int,
        signed: bool,
        path2save: Path,
    ) -> None:
        match self._settings.method:
            case DownSamplingTargets.Subsampling:
                c_compile.build_downsampling_subsampling(
                    downsampling_ratio=self._settings.dsr,
                    bitwidth=bitwidth,
                    signed=signed,
                    path2save=path2save,
                    downsampling_id=id,
                    define_path=".",
                )
            case DownSamplingTargets.Simple:
                c_compile.build_downsampling_simple(
                    downsampling_ratio=self._settings.dsr,
                    bitwidth=bitwidth,
                    signed=signed,
                    path2save=path2save,
                    downsampling_id=id,
                    define_path=".",
                )
            case DownSamplingTargets.CIC:
                c_compile.build_downsampling_cic(
                    downsampling_ratio=self._settings.dsr,
                    num_stages=self._settings.num_stages,
                    bitwidth=bitwidth,
                    signed=signed,
                    path2save=path2save,
                    downsampling_id=id,
                    define_path=".",
                )
            case DownSamplingTargets.Polyphase:
                c_compile.build_downsampling_polyphase(
                    downsampling_ratio=self._settings.dsr,
                    take_first_order=self._settings.num_stages % 2 == 1,
                    bitwidth=bitwidth,
                    signed=signed,
                    path2save=path2save,
                    downsampling_id=id,
                    define_path=".",
                )
            case _:
                raise NotImplementedError(f"Method {self._settings.method} is not implemented")

    def _create_cic_verilog(self, id: str, bitwidth: int, dec_rate: int, n_dec: int) -> dict:
        return {
            "type": "cic",
            "id": id,
            "params": {"BITWIDTH": bitwidth, "DEC_RATE": dec_rate, "N_DEC": n_dec},
        }

    def _create_polydec_fpga_verilog(self, id: str, bitwidth: int, poly_order: int) -> dict:
        return {
            "type": "polydec_fpga",
            "id": id,
            "params": {"BITWIDTH": bitwidth, "POLY_ORDER": poly_order},
        }

    def _create_polydec_asic_verilog(self, id: str, bitwidth: int, poly_order: int) -> dict:
        return {
            "type": "polydec_asic",
            "id": id,
            "params": {"BITWIDTH": bitwidth, "POLY_ORDER": poly_order},
        }

    def _create_subsampler_verilog(self, id: str, bitwidth: int, order: int) -> dict:
        return {
            "type": "subsampler",
            "id": id,
            "params": {"BITWIDTH": bitwidth, "DEC_RATE": order, "INDEX": 0},
        }

    def _create_downsampler_mean_verilog(self, id: str, bitwidth: int, order: int) -> dict:
        return {
            "type": "downsampler_mean",
            "id": id,
            "params": {"BITWIDTH": bitwidth, "DEC_RATE": order},
        }

    def _create_design_fpga_verilog(self, id: str, bitwidth: int, path2save: Path) -> None:
        match self._settings.method:
            case DownSamplingTargets.Subsampling:
                params = self._create_subsampler_verilog(
                    id=id, bitwidth=bitwidth, order=self._settings.dsr
                )
            case DownSamplingTargets.Simple:
                params = self._create_downsampler_mean_verilog(
                    id=id, bitwidth=bitwidth, order=self._settings.dsr
                )
            case DownSamplingTargets.CIC:
                params = self._create_cic_verilog(
                    id=id, bitwidth=bitwidth, dec_rate=self._settings.dsr, n_dec=self._settings.num_stages
                )
            case DownSamplingTargets.Polyphase:
                params = self._create_polydec_fpga_verilog(
                    id=id, bitwidth=bitwidth, poly_order=self._settings.dsr
                )
            case _:
                raise ValueError
        datarate_filters.load_and_plugin(packages=["datarate"], path2save=path2save, **params)

    def _create_design_asic_verilog(self, id: str, bitwidth: int, path2save: Path) -> None:
        match self._settings.method:
            case DownSamplingTargets.Subsampling:
                raise NotImplementedError
            case DownSamplingTargets.Simple:
                raise NotImplementedError
            case DownSamplingTargets.CIC:
                raise NotImplementedError
            case DownSamplingTargets.Polyphase:
                params = self._create_polydec_asic_verilog(
                    id=id, bitwidth=bitwidth, poly_order=self._settings.dsr
                )
            case _:
                raise ValueError
        datarate_filters.load_and_plugin(packages=["datarate"], path2save=path2save, **params)

    def _do_simple(self, uin: np.ndarray) -> np.ndarray:
        n = uin.size // self._settings.dsr * self._settings.dsr
        data = uin[:n]
        return data.reshape(-1, self._settings.dsr).mean(axis=1)

    def do_subsampling(self, data: np.ndarray, augment: bool = False, take_sample: int = 0) -> np.ndarray:
        """Downsample datasets by taking every dsr-th value along the last axis.
        :param data:        Numpy array with transient signal input (high sampling rate)
        :param augment:     When augment is True, additional samples are generated from the
                            remaining offsets and concatenated along the sample axis. Missing tail
                            values are zero-padded so all generated samples have equal length.
        :param take_sample: Number of samples to take
        :return:            Numpy array with transient signal output (low sampling rate)
        """
        factor = self._settings.dsr
        if factor < 1:
            raise ValueError("dsr must be >= 1")
        if factor == 1:
            return data
        if data.ndim < 1:
            raise ValueError("subsampling expects an array")

        output_length = data[..., 0::factor].shape[-1]
        downsampled_offsets = [
            self._pad_last_axis(data[..., offset::factor], output_length) for offset in range(factor)
        ]
        if not augment:
            return downsampled_offsets[take_sample]
        return np.concatenate(downsampled_offsets, axis=0)

    def _do_cic(self, uin: np.ndarray) -> np.ndarray:
        output_transient = list()
        dsr = self._settings.dsr
        gain = dsr**self._settings.num_stages

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

        intes = [integrator() for a in range(self._settings.num_stages)]
        combs = [comb() for a in range(self._settings.num_stages)]
        for s, v in enumerate(uin):
            z = round(v)
            for i in range(self._settings.num_stages):
                z = intes[i].update(z)

            if s % dsr == 0:
                for c in combs:
                    z = c.update(z)
                output_transient.append(z / gain)
        return np.array(output_transient)

    @staticmethod
    def _do_decimation_polyphase_order_one(uin: np.ndarray) -> np.ndarray:
        last_sample_hs = 0.0
        uout = list()
        for idx, val in enumerate(uin):
            if idx % 2 == 1:
                uout.append(val + last_sample_hs)
            last_sample_hs = val
        return np.array(uout)

    @staticmethod
    def _do_decimation_polyphase_order_two(uin: np.ndarray) -> np.ndarray:
        last_even_prev = 0.0
        last_even = 0.0
        uout = list()
        for idx, val in enumerate(uin):
            if idx % 2 == 0:
                last_even_prev = last_even
                last_even = val
            else:
                uout.append(val + 2 * last_even + last_even_prev)
        return np.array(uout)

    def _do_decimation_polyphase(self, uin: np.ndarray, take_first_order: bool) -> np.ndarray:
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

    def decimate(self, uin: np.ndarray) -> np.ndarray:
        """Performing the decimation filter on an input data stream
        param uin:          Numpy array with transient signal input (high sampling rate)
        return:             Numpy array with transient signal output (low sampling rate)
        """
        match self._settings.method:
            case DownSamplingTargets.Simple:
                return self._do_simple(uin)
            case DownSamplingTargets.Subsampling:
                return self.do_subsampling(data=uin, augment=False, take_sample=0)
            case DownSamplingTargets.CIC:
                return self._do_cic(uin)
            case DownSamplingTargets.Polyphase:
                return self._do_decimation_polyphase(uin, take_first_order=False)
            case _:
                raise AttributeError("Not right selected decimation method")
