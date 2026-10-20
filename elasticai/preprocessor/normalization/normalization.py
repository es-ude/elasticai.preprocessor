from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from elasticai.preprocessor._basics import DataTransient, PreprocessingModule


class DataNormalizationTargets(Enum):
    Norm = "norm"
    MedinanMAD = "medianmad"
    MeanMAD = "meanmad"
    Minmax = "minmax"
    Zscore = "zscore"
    Zeroone = "zeroone"


class DataNormalizationPeakMode(IntEnum):
    Max = 0
    Min = 1
    AbsMax = 2


@dataclass
class DataNormalizationSettings:
    """Settings for performing normalization on input data
    Attributes:
        method: TargetsNormalization [
                    Norm = "norm"
                    MedinanMAD = "medianmad"
                    MeanMAD = "meanmad"
                    Minmax = "minmax"
                    Zscore = "zscore"]
        peak_mode:  Mode for taking peak value (0: max, 1: min, 2: abs-max)
    """

    method: DataNormalizationTargets
    peak_mode: int | DataNormalizationPeakMode


DataNormalizationSettingsDefault = DataNormalizationSettings(
    method=DataNormalizationTargets.Minmax,
    peak_mode=DataNormalizationPeakMode.AbsMax,
)


class DataNormalization(PreprocessingModule[DataTransient, DataTransient]):
    _settings: DataNormalizationSettings
    __params: dict = {}

    def __init__(self, settings: DataNormalizationSettings) -> None:
        """Normalizing the input data to enhance classification performance.
        :param settings:   Settings for performing normalization on input data
        """
        super().__init__()
        self._settings = settings
        if isinstance(settings.method, str):
            self._settings.method = DataNormalizationTargets(settings.method.lower())
        self.__list_norm_methods = {
            DataNormalizationTargets.Zeroone: self._normalize_zeroone,
            DataNormalizationTargets.Minmax: self._normalize_minmax,
            DataNormalizationTargets.Norm: self._normalize_norm,
            DataNormalizationTargets.Zscore: self._normalize_zscore,
            DataNormalizationTargets.MedinanMAD: self._normalize_medianmad,
            DataNormalizationTargets.MeanMAD: self._normalize_meanmad,
        }

    def _list_normalization_methods(self) -> list:
        return [key for key in self.__list_norm_methods.keys()]

    def forward(self, x: Any) -> Any:
        if self._settings.method in self.__list_norm_methods.keys():
            return self.__list_norm_methods[self._settings.method](x)
        else:
            raise NotImplementedError("Selected mode is not available.")

    def __call__(self, x: DataTransient) -> DataTransient:
        return DataTransient(
            data=self.forward(x.data),
            sampling_rate=x.sampling_rate,
        )

    def create_design(
        self,
        id: str,
        target: str,
        path2save: Path,
        total_bitwidth: int,
        frac_bitwidth: int = 0,
        is_signed: bool = True,
    ) -> None:
        supported_targets = ["mcu", "pc", "fpga"]
        target = target.lower()
        if target not in supported_targets:
            raise ValueError(f"Target {target} is not supported: only {supported_targets}")
        if self._settings.method not in self._list_normalization_methods():
            raise ValueError(f"Method {self._settings.method} is not available!")

        if target.lower() in ["mcu", "pc"]:
            self._create_design_c(
                id=id,
                bitwidth=total_bitwidth,
                signed=is_signed,
                path2save=path2save,
            )
        else:
            self._create_design_fpga(
                id=id,
                bitwidth=total_bitwidth,
                signed=is_signed,
                path2save=path2save,
            )

    def _create_design_c(self, id: str, bitwidth: int, signed: bool, path2save: Path) -> None:
        from elasticai.creator_plugins.normalization.src import c_compile

        method = self._settings.method
        if method not in (DataNormalizationTargets.Minmax, DataNormalizationTargets.Zscore):
            raise NotImplementedError(
                "C generation currently supports only minmax and zscore normalization"
            )
        if (
            method == DataNormalizationTargets.Minmax
            and self._settings.peak_mode != DataNormalizationPeakMode.AbsMax
        ):
            raise NotImplementedError("C generation currently supports only peak_mode=AbsMax")

        builders = {
            DataNormalizationTargets.Minmax: c_compile.build_normalization_minmax,
            DataNormalizationTargets.Zscore: c_compile.build_normalization_zscore,
        }
        builders[method](
            bitwidth=bitwidth,
            signed=signed,
            path2save=path2save,
            normalization_id=id,
            define_path=".",
        )

    def _create_design_fpga(self, id: str, bitwidth: int, signed: bool, path2save: Path) -> None:
        raise NotImplementedError

    @staticmethod
    def _generate_tensor_full(data: torch.Tensor, num_repeats: int) -> torch.Tensor:
        test = torch.repeat_interleave(torch.unsqueeze(data, dim=-1), num_repeats, dim=-1)
        return test

    @staticmethod
    def _generate_numpy_full(data: np.ndarray, num_repeats: int) -> np.ndarray:
        return np.repeat(np.expand_dims(data, axis=-1), num_repeats, axis=-1)

    def _get_data_peak_value_numpy(self, raw_dataset: np.ndarray) -> np.ndarray:
        match self._settings.peak_mode:
            case 0:
                amp_array = np.max(raw_dataset, axis=-1)
            case 1:
                amp_array = np.abs(np.min(raw_dataset, axis=-1))
            case _:
                amp_array = np.max(np.abs(raw_dataset), axis=-1)
        return amp_array

    def _get_data_peak_value_tensor(self, raw_dataset: torch.Tensor) -> torch.Tensor:
        match self._settings.peak_mode:
            case 0:
                amp_array = torch.max(raw_dataset, dim=-1).values
            case 1:
                amp_array = torch.abs(torch.min(raw_dataset, dim=-1).values)
            case _:
                amp_array = torch.max(torch.abs(raw_dataset), dim=-1).values
        return amp_array

    def _get_scaling_value_minmax(self, raw_dataset: np.ndarray | torch.Tensor) -> None:
        if isinstance(raw_dataset, torch.Tensor):
            scale = self._get_data_peak_value_tensor(raw_dataset)
        else:
            scale = self._get_data_peak_value_numpy(raw_dataset)
        self.__params = {"scale_used": scale}

    ################################ IMPLEMENTED METHODS ################################
    def _normalize_zeroone(self, dataset: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        self._settings.peak_mode = DataNormalizationPeakMode.AbsMax
        self._get_scaling_value_minmax(dataset)
        if isinstance(dataset, np.ndarray):
            scale_norm = self._generate_numpy_full(2 * self.__params["scale_used"], dataset.shape[-1])
            dataset_norm = 0.5 + dataset / scale_norm
        else:
            scale_norm = self._generate_tensor_full(2 * self.__params["scale_used"], dataset.shape[-1])
            dataset_norm = torch.add(0.5, torch.divide(dataset, scale_norm))
        return dataset_norm

    def _normalize_minmax(self, dataset: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        self._settings.peak_mode = DataNormalizationPeakMode.AbsMax
        self._get_scaling_value_minmax(dataset)
        if isinstance(dataset, np.ndarray):
            scale_norm = self._generate_numpy_full(self.__params["scale_used"], dataset.shape[-1])
            dataset_norm = dataset / scale_norm
        else:
            scale_norm = self._generate_tensor_full(self.__params["scale_used"], dataset.shape[-1])
            dataset_norm = torch.divide(dataset, scale_norm)
        return dataset_norm

    def _get_scaling_value_norm(self, raw_dataset: np.ndarray | torch.Tensor) -> None:
        if isinstance(raw_dataset, np.ndarray):
            scale = np.linalg.norm(raw_dataset, axis=-1)
        else:
            scale = torch.norm(raw_dataset, dim=-1)
        self.__params = {"scale_used": scale}

    def _normalize_norm(self, dataset: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        self._get_scaling_value_norm(dataset)
        if isinstance(dataset, np.ndarray):
            scale_norm = self._generate_numpy_full(self.__params["scale_used"], dataset.shape[-1])
            dataset_norm = dataset / scale_norm
        else:
            scale_norm = self._generate_tensor_full(self.__params["scale_used"], dataset.shape[-1])
            dataset_norm = torch.divide(dataset, scale_norm)
        return dataset_norm

    def _get_scaling_value_zscore(self, raw_dataset: np.ndarray | torch.Tensor) -> None:
        scale_std = (
            np.std(raw_dataset, axis=-1)
            if isinstance(raw_dataset, np.ndarray)
            else torch.std(raw_dataset, dim=-1, unbiased=False)
        )
        scale_mean = (
            np.mean(raw_dataset, axis=-1)
            if isinstance(raw_dataset, np.ndarray)
            else torch.mean(raw_dataset, dim=-1)
        )
        self.__params = {"scale_std": scale_std, "scale_mean": scale_mean}

    def _normalize_zscore(self, dataset: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        self._get_scaling_value_zscore(dataset)
        if isinstance(dataset, np.ndarray):
            scale_mean = self._generate_numpy_full(self.__params["scale_mean"], dataset.shape[-1])
            scale_std = self._generate_numpy_full(self.__params["scale_std"], dataset.shape[-1])
            dataset_norm = (dataset - scale_mean) / scale_std
        else:
            scale_mean = self._generate_tensor_full(self.__params["scale_mean"], dataset.shape[-1])
            scale_std = self._generate_tensor_full(self.__params["scale_std"], dataset.shape[-1])
            dataset_norm = torch.divide(torch.sub(dataset, scale_mean), scale_std)

        self.__params["scale_used"] = scale_mean / scale_std
        return dataset_norm

    def _get_scaling_value_medianmad(self, raw_dataset: np.ndarray | torch.Tensor) -> None:
        if isinstance(raw_dataset, np.ndarray):
            scale_median = np.median(raw_dataset, axis=-1)
            scale_mad = np.median(
                np.abs(raw_dataset - self._generate_numpy_full(scale_median, raw_dataset.shape[-1])),
                axis=-1,
            )
        else:
            scale_median = torch.quantile(raw_dataset, 0.5, dim=-1)
            scale_mad = torch.quantile(
                torch.abs(raw_dataset - self._generate_tensor_full(scale_median, raw_dataset.shape[-1])),
                0.5,
                dim=-1,
            )
        self.__params = {"scale_mad": scale_mad, "scale_median": scale_median}

    def _normalize_medianmad(self, dataset: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        self._get_scaling_value_medianmad(dataset)
        if isinstance(dataset, np.ndarray):
            scale_median = self._generate_numpy_full(self.__params["scale_median"], dataset.shape[-1])
            scale_mad = self._generate_numpy_full(self.__params["scale_mad"], dataset.shape[-1])
            dataset_norm = (dataset - scale_median) / scale_mad
        else:
            scale_median = self._generate_tensor_full(self.__params["scale_median"], dataset.shape[-1])
            scale_mad = self._generate_tensor_full(self.__params["scale_mad"], dataset.shape[-1])
            dataset_norm = torch.divide(torch.sub(dataset, scale_median), scale_mad)

        self.__params["scale_used"] = scale_median / scale_mad
        return dataset_norm

    def _get_scaling_value_meanmad(self, raw_dataset: np.ndarray | torch.Tensor) -> None:
        if isinstance(raw_dataset, np.ndarray):
            scale_mean = np.mean(raw_dataset, axis=-1)
            scale_mad = np.mean(
                np.abs(raw_dataset - self._generate_numpy_full(scale_mean, raw_dataset.shape[-1])),
                axis=-1,
            )
        else:
            scale_mean = torch.mean(raw_dataset, dim=-1)
            scale_mad = torch.mean(
                torch.abs(raw_dataset - self._generate_tensor_full(scale_mean, raw_dataset.shape[-1])),
                dim=-1,
            )
        self.__params = {"scale_mad": scale_mad, "scale_mean": scale_mean}

    def _normalize_meanmad(self, dataset: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        self._get_scaling_value_meanmad(dataset)
        if isinstance(dataset, np.ndarray):
            scale_mean = self._generate_numpy_full(self.__params["scale_mean"], dataset.shape[-1])
            scale_mad = self._generate_numpy_full(self.__params["scale_mad"], dataset.shape[-1])
            dataset_norm = (dataset - scale_mean) / scale_mad
        else:
            scale_mean = self._generate_tensor_full(self.__params["scale_mean"], dataset.shape[-1])
            scale_mad = self._generate_tensor_full(self.__params["scale_mad"], dataset.shape[-1])
            dataset_norm = torch.divide(torch.sub(dataset, scale_mean), scale_mad)

        self.__params["scale_used"] = scale_mean / scale_mad
        return dataset_norm
