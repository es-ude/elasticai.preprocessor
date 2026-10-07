from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams

from elasticai.preprocessor.sequential import BuildPlatformTargets, PreprocessingModule, SequentialSignal


class RescalerTargets(Enum):
    FxP = "fxp"
    IntSym = "int_sym"
    IntAsym = "int_asym"


@dataclass
class RescalerSettings:
    """"""

    method: RescalerTargets


RescalerSettingsDefault = RescalerSettings(
    method=RescalerTargets.FxP,
)


class Rescaler(PreprocessingModule):
    _settings: RescalerSettings
    _arith: FxpArithmetic
    _allowed_input_types = ["int"]

    def __init__(self, settings: RescalerSettings) -> None:
        """Class for resampling transient input data to model desired data stream output (digital only)
        :param settings:    Settings for defining the properties of the data rescaling method
        :return:            None
        """
        super().__init__()
        self._settings = settings
        # TODO: Change this one
        self._arith = FxpArithmetic(FxpParams(total_bits=8, frac_bits=4, signed=True))
        if isinstance(settings.method, str):
            self._settings.method = RescalerTargets(settings.method)

    def __call__(self, x: SequentialSignal) -> SequentialSignal:
        return SequentialSignal(
            data=self.convert(x.data),
            sample_rate=x.sample_rate,
        )

    def _clamp_digital(self, data: np.ndarray, use_integer: bool) -> np.ndarray:
        if use_integer:
            return np.clip(
                a=data, a_min=self._arith.minimum_as_integer, a_max=self._arith.maximum_as_integer
            )
        else:
            return np.clip(
                a=data, a_min=self._arith.minimum_as_rational, a_max=self._arith.maximum_as_rational
            )

    def _quantize(self, data: np.ndarray, is_int_output: bool) -> np.ndarray:
        def _get_dtype(total_bits: int, is_signed: bool) -> np.dtype:
            for bits in (8, 16, 32, 64):
                if total_bits <= bits:
                    return np.dtype(f"{'int' if is_signed else 'uint'}{bits}")
            raise AttributeError(f"Unknown datatype for total_bits = {total_bits}")

        if is_int_output:
            xout = [self._arith.round_to_integer(val) for val in data]
            shape = _get_dtype(total_bits=self._settings.total_bits, is_signed=self._settings.is_signed)
        else:
            xout = [self._arith.round_to_rational(val) for val in data]
            shape = np.float32
        return np.asarray(xout, dtype=shape)

    def _quantize_digital(self, data: np.ndarray, is_int_input: bool, is_int_output: bool) -> np.ndarray:
        xin = self._clamp_digital(data=data, use_integer=is_int_input)
        if is_int_input:
            xin = (self._arith._config.minimum_step_as_rational * xin).tolist()
        else:
            xin = xin.tolist()
        return self._quantize(data=xin, is_int_output=is_int_output)

    def convert(self, x: np.ndarray) -> np.ndarray:
        """Function for converting input data stream to desired data stream output
        :param x:       Input data stream
        :return:        Output data stream
        """
        if "int" not in str(x.dtype):
            raise AttributeError("Input data must be an integer type (simulating an ADC output directly)")
        return x

    def create_design(self, target: str | BuildPlatformTargets, id: str, path2save: Path) -> None:
        """Function for creating the hardware design to process raw data in streaming processes
        :param target:      Target for hardware design (MCU, FPGA, PC)
        :param id:          ID of hardware designs
        :param path2save:   Path to the saved hardware designs
        :return:            None
        """
        if isinstance(target, str):
            target0 = BuildPlatformTargets(target.lower())
        else:
            target0 = target
        if target0 in [BuildPlatformTargets.Workstation, BuildPlatformTargets.MCU]:
            self._create_design_c(
                id=id,
                path2save=path2save,
                define_path="src",
            )
        else:
            self._create_design_verilog(id=id, path2save=path2save)

    def _create_design_c(self, id: str, path2save: Path, define_path: str = "src"):
        from elasticai.creator_plugins.adc.src import c_compile

        raise NotImplementedError
        c_compile.build_adc_quant(
            adc_id=id,
            define_path=define_path,
            settings=self._settings,
            path2save=path2save,
        )

    def _create_design_verilog(self, id: str, path2save: Path) -> None:
        raise NotImplementedError("FPGA design is not implemented yet")
