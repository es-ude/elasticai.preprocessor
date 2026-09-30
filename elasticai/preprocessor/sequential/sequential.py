from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class TargetsBuildPlatform(Enum):
    Workstation = "pc"
    MCU = "mcu"
    FPGA = "fpga"
    ASIC = "asic"


@dataclass
class SettingsCreateSequential:
    """Settings for building the pipeline segments for hardware platform

    Attributes:
        target:         Definition of the hardware target to build (supported TargetsBuildPlatform)
        total_bitwidth: Integer with the total bitwidth for representing the data
        frac_bitwidth:  Integer with the fractional bitwidth for representing the data (if fixed_point)
        do_signed:      Boolean for setting the signed data format
        path2build:     Path to build output directory
    """

    target: TargetsBuildPlatform
    total_bitwidth: int
    frac_bitwidth: int
    do_signed: bool
    path2build: Path


@dataclass
class SequentialSignal:
    """Dataclass for handling the transient data inside the pipeline
    Attributes:
        data:           Numpy array of the data
        sample_rate:    Float with sampling rate [Hz]
    """

    data: np.ndarray
    sample_rate: float


class PreprocessingModule(ABC):
    @abstractmethod
    def __call__(self, x: SequentialSignal) -> SequentialSignal: ...

    @abstractmethod
    def create_design(self, id: str, settings: SettingsCreateSequential) -> None: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class PreprocessingSequential:
    modules: list[PreprocessingModule]

    def __init__(self, *modules: PreprocessingModule):
        self.modules = list(modules)

    def __call__(self, x: np.ndarray, fs: float) -> SequentialSignal:
        if not len(self):
            raise AttributeError("No pipelines are loaded")

        data = SequentialSignal(
            data=x,
            sample_rate=fs,
        )
        for module in self.modules:
            data = module(data)
        return data

    def __getitem__(self, idx):
        return self.modules[idx]

    def __len__(self):
        return len(self.modules)

    def append(self, module: PreprocessingModule):
        self.modules.append(module)
        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        inner = ",\n\t".join(repr(m) for m in self.modules)
        match len(self):
            case 0:
                return f"{class_name}()"
            case _:
                return f"{class_name}(\n\t{inner}\n)"

    def create_design(self, settings: SettingsCreateSequential) -> None:
        """Create the pipeline design for deploying on hardware
        :param settings:    Dataclass SettingsCreateSequential for building the hardware designs
        :return:            None
        """
        if not len(self):
            raise AttributeError("No pipelines are loaded")
        for idx, module in enumerate(self.modules):
            if not hasattr(module, "create_design"):
                raise AttributeError(f"module {module} has no `create_design` method")
            module.create_design(id=f"{idx}", settings=settings)
