from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic

import numpy as np
from typing_extensions import TypeVar


def get_path_to_project(new_folder: str = "") -> Path:
    """Function for getting the path to find the project folder structure in application.
    :param new_folder:  New folder path
    :return:            String of absolute path to start the project structure
    """
    max_levels = 5
    current = Path("..").absolute()

    def is_project_root(p: Path) -> bool:
        return (p / "pyproject.toml").exists()

    for _ in range(max_levels):
        if is_project_root(current):
            continue
        current = current.parent
    if new_folder:
        return current.absolute() / new_folder
    else:
        return current.absolute()


@dataclass
class DataTransient:
    """Dataclass for handling the transient data inside the pipeline
    Attributes:
        data:           Numpy array of the data
        sampling_rate:  Float with sampling rate [Hz]
    """

    data: np.ndarray | Any
    sampling_rate: float


@dataclass
class DataSpectral:
    """Dataclass for handling the spectral data inside the pipeline
    Attributes:
        data:           Numpy array of the data
        sampling_rate:  Float with sampling rate [Hz]
    """

    data: np.ndarray
    freq: np.ndarray
    sampling_rate: float


@dataclass
class DataWaveform:
    """Dataclass for handling the waveform data inside the pipeline
    Attributes:
        waveform:       Numpy array of the data
        xpos:           Numpy array with positions which an event is detected
        label:          Numpy array with labels for each event
        sampling_rate:  Float with sampling rate [Hz]
    """

    waveform: np.ndarray
    xpos: np.ndarray
    label: np.ndarray
    sampling_rate: float

    @property
    def length(self) -> int:
        return self.waveform.shape[1]

    @property
    def num_samples(self) -> int:
        return self.xpos.size

    @property
    def is_data_labeled(self) -> bool:
        return np.unique(self.label).size > 0 and 255 not in self.label.tolist()


TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class PreprocessingModule(ABC, Generic[TIn, TOut]):
    @abstractmethod
    def __call__(self, x: TIn) -> TOut:
        """"""
        ...

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward function for processing the input data stream to new output format
        :param x:   Numpy array with input data stream
        :return:    Numpy array with output data stream
        """
        ...

    def __repr__(self) -> str:
        """"""
        return f"{self.__class__.__name__}"

    @abstractmethod
    def create_design(
        self,
        id: str,
        target: str,
        path2save: Path,
        total_bitwidth: int,
        frac_bitwidth: int,
        is_signed: bool,
    ) -> None:
        """Function for creating the hardware design of this processing module
        :param id:              String with ID of this hardware design
        :param target:          Target for hardware design (PC, MCU, FPGA, ASIC)
        :param path2save:       Path to the saved hardware designs
        :param total_bitwidth:  Integer with total bitwidth of each value for processing
        :param frac_bitwidth:   Integer with fraction bitwidth if FxP is applied
        :param is_signed:       Boolean if signed integers is used
        :return:                None
        """
        ...
