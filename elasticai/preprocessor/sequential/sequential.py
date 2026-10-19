from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from elasticai.preprocessor._basics import DataTransient, DataWaveform, PreprocessingModule


class BuildPlatformTargets(Enum):
    Workstation = "pc"
    MCU = "mcu"
    FPGA = "fpga"
    ASIC = "asic"


@dataclass
class CreateSequentialSettings:
    """Settings for building the pipeline segments for hardware platform

    Attributes:
        target:         Definition of the hardware target to build (supported TargetsBuildPlatform)
        total_bitwidth: Integer with the total bitwidth for representing the data
        frac_bitwidth:  Integer with the fractional bitwidth for representing the data (if fixed_point)
        do_signed:      Boolean for setting the signed data format
        path2build:     Path to build output directory
    """

    target: BuildPlatformTargets
    total_bitwidth: int
    frac_bitwidth: int
    do_signed: bool
    path2build: Path


class PreprocessingSequential:
    modules: list[PreprocessingModule]

    def __init__(self, *modules: PreprocessingModule):
        self.modules = list(modules)

    def __call__(self, x: np.ndarray, fs: float) -> DataTransient | DataWaveform:
        if not len(self):
            raise AttributeError("No pipelines are loaded")

        data = DataTransient(
            data=x,
            sampling_rate=fs,
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

    def create_design(self, settings: CreateSequentialSettings) -> None:
        """Create the pipeline design for deploying on hardware
        :param settings:    Dataclass SettingsCreateSequential for building the hardware designs
        :return:            None
        """
        if not len(self):
            raise AttributeError("No pipelines are loaded")

        self._create_sequential(target=settings.target, path2save=settings.path2build)
        path2src = settings.path2build / "src"
        path2src.mkdir(exist_ok=True, parents=True)
        for idx, module in enumerate(self.modules):
            if not hasattr(module, "create_design"):
                raise AttributeError(f"module {module} has no `create_design` method")
            module.create_design(
                id=f"{idx}",
                target=settings.target.value,
                path2save=path2src,
                total_bitwidth=settings.total_bitwidth,
                frac_bitwidth=settings.frac_bitwidth,
                is_signed=settings.do_signed,
            )

    def _create_sequential(self, target: BuildPlatformTargets, path2save: Path) -> None:
        match target:
            case BuildPlatformTargets.Workstation:
                self._create_sequential_c(path2save=path2save)
            case BuildPlatformTargets.MCU:
                self._create_sequential_c(path2save=path2save)
            case BuildPlatformTargets.FPGA:
                self._create_sequential_verilog(path2save=path2save)
            case BuildPlatformTargets.ASIC:
                self._create_sequential_asic(path2save=path2save)
            case _:
                raise AttributeError

    def _create_sequential_asic(self, path2save: Path) -> None:
        pass

    def _create_sequential_verilog(self, path2save: Path) -> None:
        pass

    def _create_sequential_c(self, path2save: Path) -> None:
        pass
