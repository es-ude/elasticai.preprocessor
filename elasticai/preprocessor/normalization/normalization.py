from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class SettingsNormalization:
    """Settings for performing normalization on input data
    Attributes:
        method (str):               The normalization method ["minmax", "norm", "zscore", "medianmad", or "meanmad"]
        peak_mode (int):            Mode for taking peak value (0: max, 1: min, 2: abs-max)

    """

    method: str
    peak_mode: int


DefaultSettingsNormalization = SettingsNormalization(
    method="minmax",
    peak_mode=2,
)


class DataNormalization:
    _settings: SettingsNormalization
    __params: dict = {}

    def __init__(self, settings: SettingsNormalization):
        """Normalizing the input data to enhance classification performance.
        Parameters:
            settings:   Settings for performing normalization on input data
        Methods:
            normalize(): Normalize the input data based on the selected mode and method.
        Examples:
            # Create an instance of DataNormalization
            handler = DataNormalization("minmax")
            data_in = (0.5 - np.random.rand(100, 10)) * 10
            normalized_frames = handler.normalize(data_in)
        """
        self._settings = settings
        self.__list_norm_methods = {
            "zeroone": self._normalize_zeroone,
            "minmax": self._normalize_minmax,
            "norm": self._normalize_norm,
            "zscore": self._normalize_zscore,
            "medianmad": self._normalize_medianmad,
            "meanmad": self._normalize_meanmad,
        }

    def list_normalization_methods(self) -> list:
        """Return list with all available methods for normalization"""
        return [key for key in self.__list_norm_methods.keys()]

    def get_peak_amplitude_values(self) -> np.ndarray | torch.Tensor:
        """Getting the peak amplitude of rawdata as array"""
        key_search = "scale_used"
        if key_search in self.__params.keys():
            return self.__params[key_search]
        else:
            raise NotImplementedError("Key scale_local is not available!")

    def normalize(self, dataset: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Apply normalization methods on input data
        Args:
            Numpy array with frames for normalizing
        Returns:
            Numpy array with normalized frames
        """
        if self._settings.method.lower() in self.__list_norm_methods.keys():
            return self.__list_norm_methods[self._settings.method.lower()](dataset)
        else:
            raise NotImplementedError("Selected mode is not available.")

    def create_design(
        self,
        target: str,
        bitwidth: int,
        id: str,
        path2save: Path,
        signed: bool = True,
    ) -> None:
        """Generate a C design for the configured normalization method.
        :param target:      String with target name ["mcu", "pc", "fpga"]
        :param bitwidth:    Integer with total bitwidth
        :param id:          String with unique identifier of device (appended to the name)
        :param path2save:   Path to save the hardware files
        :param signed:      Whether generated C designs use a signed integer data type
        :return:            None
        """
        supported_targets = ["mcu", "pc", "fpga"]
        target = target.lower()
        if target not in supported_targets:
            raise ValueError(f"Target {target} is not supported: only {supported_targets}")
        if self._settings.method.lower() not in self.list_normalization_methods():
            raise ValueError(f"Method {self._settings.method.lower()} is not available!")

        if target.lower() in ["mcu", "pc"]:
            self._create_design_c(
                id=id,
                bitwidth=bitwidth,
                signed=signed,
                path2save=path2save,
            )
        else:
            self._create_design_fpga(
                id=id,
                bitwidth=bitwidth,
                signed=signed,
                path2save=path2save,
            )

    def _create_design_c(self, id: str, bitwidth: int, signed: bool, path2save: Path) -> None:
        from elasticai.creator_plugins.normalization.src import c_compile

        method = self._settings.method.lower()
        if method not in ("minmax", "zscore"):
            raise NotImplementedError(
                "C generation currently supports only minmax and zscore normalization"
            )
        if method == "minmax" and self._settings.peak_mode != 2:
            raise NotImplementedError("C generation currently supports only peak_mode=2")

        builders = {
            "minmax": c_compile.build_normalization_minmax,
            "zscore": c_compile.build_normalization_zscore,
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
        self._get_scaling_value_minmax(dataset)
        if isinstance(dataset, np.ndarray):
            scale_norm = self._generate_numpy_full(2 * self.__params["scale_used"], dataset.shape[-1])
            dataset_norm = 0.5 + dataset / scale_norm
        else:
            scale_norm = self._generate_tensor_full(2 * self.__params["scale_used"], dataset.shape[-1])
            dataset_norm = torch.add(0.5, torch.divide(dataset, scale_norm))
        return dataset_norm

    def _normalize_minmax(self, dataset: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
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
