from dataclasses import dataclass
from enum import Enum
from logging import Logger, getLogger
from pathlib import Path
from typing import Any

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
    _logger: Logger
    modules: list[PreprocessingModule]

    def __init__(self, *modules: PreprocessingModule):
        self._logger = getLogger(__name__)
        self.modules = list(modules)

    def __call__(self, x: np.ndarray, fs: float) -> Any:
        def prozess_data(x: np.ndarray, fs: float) -> DataTransient | DataWaveform:
            data = DataTransient(
                data=x,
                sampling_rate=fs,
            )
            for module in self.modules:
                data = module(data)
            return data

        if not len(self):
            raise AttributeError("No pipelines are loaded")

        if len(x.shape) == 1:
            self._logger.debug("Data is only single channel")
            return prozess_data(x=x, fs=fs)
        elif len(x.shape) == 2:
            if x.shape[1] <= x.shape[0]:
                raise ValueError("Please adapt the input data structure to (num_channels, num_samples)")
            self._logger.debug("Data is multi-channel")

            data0 = list()
            fs0 = list()
            for xraw in x:
                out = prozess_data(
                    x=xraw,
                    fs=fs,
                )
                data0.append(out.data)
                fs0.append(out.sampling_rate)
            return DataTransient(data=np.asarray(data0), sampling_rate=fs0[-1])

    def __getitem__(self, idx):
        return self.modules[idx]

    def __len__(self):
        return len(self.modules)

    def append(self, module: PreprocessingModule):
        self.modules.append(module)
        self._logger.debug(f"Add {module} to processing pipeline")
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

        self._logger.debug("\nCreate hardware designs")
        self._create_sequential(settings=settings, num_inputs=1, num_outputs=1)
        path2src = settings.path2build / "src"
        path2src.mkdir(exist_ok=True, parents=True)
        for idx, module in enumerate(self.modules):
            if not hasattr(module, "create_design"):
                raise AttributeError(f"module {module} has no `create_design` method")
            self._logger.debug(f"\nCreate hardware design for {module}")
            module.create_design(
                id=f"{idx}",
                target=settings.target.value,
                path2save=path2src,
                total_bitwidth=settings.total_bitwidth,
                frac_bitwidth=settings.frac_bitwidth,
                is_signed=settings.do_signed,
            )

    def _create_sequential(
        self, settings: CreateSequentialSettings, num_inputs: int, num_outputs: int
    ) -> None:
        self._logger.debug("\nCreate sequential for doing the pre-processing")
        if isinstance(settings.target, str):
            settings.target = BuildPlatformTargets(settings.target.lower())
        else:
            settings.target = settings.target

        match settings.target:
            case BuildPlatformTargets.Workstation:
                self._create_sequential_c(
                    settings=settings, num_inputs=num_inputs, num_outputs=num_outputs
                )
            case BuildPlatformTargets.MCU:
                self._create_sequential_c(
                    settings=settings, num_inputs=num_inputs, num_outputs=num_outputs
                )
            case BuildPlatformTargets.FPGA:
                self._create_sequential_verilog(
                    settings=settings, num_inputs=num_inputs, num_outputs=num_outputs
                )
            case BuildPlatformTargets.ASIC:
                self._create_sequential_verilog(
                    settings=settings, num_inputs=num_inputs, num_outputs=num_outputs
                )
            case _:
                raise AttributeError

    @staticmethod
    def _create_sequential_verilog(
        settings: CreateSequentialSettings, num_inputs: int, num_outputs: int
    ) -> None:
        from elasticai.creator_plugins.sequential.utils import load_and_plugin

        load_and_plugin(
            type="sequential",
            id="0",
            params={
                "BITWIDTH_INPUT": settings.total_bitwidth,
                "BITWIDTH_OUTPUT": settings.total_bitwidth,
                "SIZE_INPUT": num_inputs,
                "SIZE_OUTPUT": num_outputs,
            },
            path2save=settings.path2build,
        )

    @staticmethod
    def _create_sequential_c(
        settings: CreateSequentialSettings, num_inputs: int, num_outputs: int
    ) -> None:
        from elasticai.creator_plugins.sequential.src.sequential_c import build_sequential

        build_sequential(
            seq_id="PREPROCESSING_0",
            path2save=settings.path2build,
            bitwidth_input=settings.total_bitwidth,
            bitwidth_output=settings.total_bitwidth,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            is_signed=settings.do_signed,
        )
