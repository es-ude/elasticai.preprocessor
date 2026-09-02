from dataclasses import dataclass
from enum import Enum
from logging import Logger, getLogger
from pathlib import Path

import numpy as np
from scipy.signal import iirfilter, lfilter
from elasticai.creator_plugins.eventdetection.src import c_compile


class TargetsEventPreprocessors(Enum):
    Normal = "normal"
    Absolute = "absolute"
    NEO = "neo"
    MTEO = "mteo"
    ADO = "ado"
    ASO = "aso"
    EED = "eed"
    SBP = "sbp"


@dataclass
class SettingsEventPreprocessor:
    """Configuration class for defining the Spike Detection Algorithm (SDA)
    Attributes:
        type:           Applied comparator preprocessing method for transient signals [normal, absolute, Non-Linear Energy Operator (NEO) or Teager-Kaiser-Operator (window_size = 1 or kNEO with window_size > 1),
                        Multiresolution Teager Energy Operator (MTEO), absolute difference operator (ADO),
                        enhanced energy-derivation operator (eED),
                        amplitude slope operator (ASO, window_size and f_hp as additional float arg),
                        spike band-power estimation (SBP, using f_bp with two values as additional arg)
        sampling_rate:  Sampling rate [Hz]
        window_size:    Position difference for extracting SDA method. Configuration with length(x) == 1: with dX = 1 --> NEO, dX > 1 --> k-NEO
        f_filt:         List with filter frequencies for the methods (ASO, SBP)
    """

    type: TargetsEventPreprocessors
    sampling_rate: float
    window_size: list[int]
    f_filt: list[float]


DefaultSettingsEventPreprocessor = SettingsEventPreprocessor(
    type=TargetsEventPreprocessors.Normal, 
    sampling_rate=10e3, 
    window_size=[5], 
    f_filt=[150.0]
)


class EventPreprocessor:
    _logger: Logger
    _settings: SettingsEventPreprocessor

    def __init__(self, settings: SettingsEventPreprocessor) -> None:
        """Class for performing the comparator preprocessing for transient signals
        :param settings:    Settings object for defining the preprocessor
        :return:            None"""
        self._logger = getLogger(__name__)
        self._settings = settings
        if isinstance(settings.type, str):
            self._settings.type = TargetsEventPreprocessors(settings.type)

    def _get_methods(self) -> list:
        split_key = "_sda_"
        return [method.split(split_key)[-1] for method in dir(self) if split_key in method]

    @staticmethod
    def _sda_normal(xin: np.ndarray) -> np.ndarray:
        return xin

    def _sda_absolute(self, xin: np.ndarray) -> np.ndarray:
        return np.absolute(xin)

    def _sda_neo(self, xin: np.ndarray) -> np.ndarray:
        ksda0 = self._settings.window_size[0]
        x_neo0 = xin[ksda0:-ksda0] ** 2 - xin[: -2 * ksda0] * xin[2 * ksda0 :]
        return np.concatenate([x_neo0[:ksda0,], x_neo0, x_neo0[-ksda0:,]], axis=None)

    def _sda_mteo(self, xin: np.ndarray) -> np.ndarray:
        x_mteo = np.zeros(shape=(len(self._settings.window_size), xin.size))
        for idx, ksda0 in enumerate(self._settings.window_size):
            x0 = np.power(xin[ksda0:-ksda0,], 2) - xin[: -2 * ksda0,] * xin[2 * ksda0 :,]
            x_mteo[idx, :] = np.concatenate([x0[:ksda0,], x0, x0[-ksda0:,]], axis=None)
        return np.max(x_mteo, axis=0)

    def _sda_ado(self, xin: np.ndarray) -> np.ndarray:
        ksda0 = self._settings.window_size[0]
        x_sda = np.absolute(xin[ksda0:,] - xin[:-ksda0,])
        return np.concatenate([x_sda[:ksda0], x_sda], axis=None)

    def _sda_aso(self, xin: np.ndarray) -> np.ndarray:
        ksda0 = self._settings.window_size[0]
        x_sda = xin[ksda0:,] * (xin[ksda0:,] - xin[:-ksda0,])
        return np.concatenate([x_sda[:ksda0], x_sda], axis=None)

    def _sda_eed(self, xin: np.ndarray) -> np.ndarray:
        filter = iirfilter(
            N=2,
            Wn=2 * self._settings.f_filt[0] / self._settings.sampling_rate,
            ftype="butter",
            btype="highpass",
            analog=True,
            output="ba",
        )
        return np.square(np.array(lfilter(filter[0], filter[1], xin)))

    def _sda_sbp(self, xin: np.ndarray) -> np.ndarray:
        filter = iirfilter(
            N=2,
            Wn=2 * np.array(self._settings.f_filt) / self._settings.sampling_rate,
            ftype="butter",
            btype="bandpass",
            analog=False,
            output="ba",
        )
        filt0 = lfilter(filter[0], filter[1], xin)
        return np.abs(filt0)

    def get_preprocessed(self, xraw: np.ndarray) -> np.ndarray:
        """Returning the transient signal of the preprocessed comparator values
        :param xraw:    Raw signal of the input signal
        :return:        Transient signal of the preprocessed comparator values
        """
        if len(self._settings.window_size) < 1:
            raise ValueError("Length of dx_sda must be greater than 1")
        if self._settings.window_size[0] < 1:
            raise ValueError("Value of dx_sda[0] must be greater than 1")
        if self._settings.type.value not in self._get_methods():
            raise ValueError(
                f"Event Preprocessing Method '{self._settings.type}' is not known. Please change!"
            )
        return getattr(self, f"_sda_{self._settings.type.value}")(xraw)

    def create_design(
        self,
        id: str,
        target: str,
        bitwidth: int,
        signed: bool,
        path2save: Path,
    ) -> None:
        """Generate hardware design files for SDA-Preprocessing.
        :param id:        ID appended to generated function names.
        :param target:    Target platform ["mcu", "pc"].
        :param bitwidth:  Bitwidth of each sample.
        :param signed:    True if the data type is signed.
        :param path2save: Path to save the generated files.
        :param thr_val:   Constant threshold value (integer, already quantized).
        :param method:    Preprocessing method
        """
        supported_targets = ["mcu", "pc"]
        if target.lower() not in supported_targets:
            raise ValueError(f"Target '{target}' is not supported: only {supported_targets}")

        self._create_design_c(
            id=id,
            bitwidth=bitwidth,
            signed=signed,
            path2save=path2save,
        )

    def _create_design_c(
        self,
        id: str,
        bitwidth: int,
        signed: bool,
        path2save: Path,

    ) -> None:
        match self._settings.type:
            case TargetsEventPreprocessors.Normal:
                c_compile.build_preprocessor_normal(
                    bitwidth=bitwidth,
                    signed=signed,
                    path2save=path2save,
                    preprocessor_id=id,
                    define_path=".",
                )
            case TargetsEventPreprocessors.Absolute:
                c_compile.build_preprocessor_abs(
                    bitwidth=bitwidth,
                    signed=signed,
                    path2save=path2save,
                    preprocessor_id=id,
                    define_path=".",
                )
            case TargetsEventPreprocessors.NEO:
                c_compile.build_preprocessor_neo(
                    bitwidth=bitwidth,
                    signed=signed,
                    path2save=path2save,
                    preprocessor_id=id,
                    define_path=".",
                )
            case _:
                raise NotImplementedError(
                    f"C desing for preprocessor_type={self._settings.type}, is not implemented yet."
                )
