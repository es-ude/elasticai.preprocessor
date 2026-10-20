from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pytest
import torch

from .normalization import (
    DataNormalization,
    DataNormalizationSettings,
    DataNormalizationTargets,
    DataTransient,
)


def generate_test_data(
    num_samples: int = 100, num_window_size: int = 32, do_tensor: bool = False
) -> np.ndarray | torch.Tensor:
    """Test function for generating test data
    :param num_samples:         Number of samples to generate
    :param num_window_size:     Number of window size
    :param do_tensor:           Boolean if output type is tensor
    :return:                    Tensor or numpy array with test data
    """
    if do_tensor:
        x = torch.linspace(0, 2 * torch.pi, num_window_size)
        data = torch.zeros((num_samples, num_window_size))
        for idx in range(num_samples):
            data[idx, :] = 10 * torch.sin(x - idx / num_samples * torch.pi)
    else:
        x = np.linspace(0, 2 * np.pi, num_window_size)
        data = np.zeros((num_samples, num_window_size))
        for idx in range(num_samples):
            data[idx, :] = 10 * np.sin(x - idx / num_samples * np.pi)
    return data


def generate_reference_array(data_array: np.ndarray | list, index_array: str = "check") -> str:
    """Function for generating a reference array to load into the testbench
    :param data_array:  numpy array or list with data
    :param index_array: index name of data array for testing
    :return:            string with index name and list array with data
    """
    if type(data_array) == np.ndarray:
        list_to_transfer = data_array.tolist()
    else:
        list_to_transfer = data_array
    list_to_transfer = [str(val) for val in list_to_transfer]

    string_out = f"{index_array} = [" + ", ".join(list_to_transfer) + "]"
    return string_out


class TestHelper(TestCase):
    num_input = 10
    num_window_size = 100

    def test_generate_reference_list_value(self):
        test_data = [idx for idx in range(10)]
        result = generate_reference_array(test_data, "check")
        local_vars = {}
        exec(result, {}, local_vars)

        np.testing.assert_array_equal(np.array(test_data), np.array(local_vars["check"]))

    def test_generate_reference_integer(self):
        test_data = [idx for idx in range(10)]
        result = generate_reference_array(test_data, "check")
        check = "check = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"
        np.testing.assert_array_equal(result, check)

    def test_generate_reference_float(self):
        test_data = np.linspace(-1, +1, 11, endpoint=True, dtype=float)
        result = generate_reference_array(test_data, "check")
        check = "check = [-1.0, -0.8, -0.6, -0.3999999999999999, -0.19999999999999996, 0.0, 0.20000000000000018, 0.40000000000000013, 0.6000000000000001, 0.8, 1.0]"
        np.testing.assert_array_equal(result, check)

    def test_generate_testdata_type_tensor(self):
        test_data = generate_test_data(
            num_samples=self.num_input,
            num_window_size=self.num_window_size,
            do_tensor=True,
        )
        self.assertEqual(test_data.dtype, torch.float32)

    def test_generate_testdata_shape_tensor(self):
        test_data = generate_test_data(
            num_samples=self.num_input,
            num_window_size=self.num_window_size,
            do_tensor=True,
        )
        self.assertEqual(test_data.shape, torch.Size([self.num_input, self.num_window_size]))

    def test_generate_testdata_type_numpy(self):

        test_data = generate_test_data(
            num_samples=self.num_input,
            num_window_size=self.num_window_size,
            do_tensor=False,
        )
        self.assertEqual(test_data.dtype, np.float64)

    def test_generate_testdata_shape_numpy(self):
        test_data = generate_test_data(
            num_samples=self.num_input,
            num_window_size=self.num_window_size,
            do_tensor=False,
        )
        self.assertEqual(test_data.shape, (self.num_input, self.num_window_size))


def test_list_methods():
    sets = DataNormalizationSettings(
        method="minmax",
        peak_mode=0,
    )
    test_func = DataNormalization(settings=sets)
    key = test_func._list_normalization_methods()
    assert len(key) == 6


def test_error_wrong_input():
    with pytest.raises(ValueError):
        sets = DataNormalizationSettings(
            method="bimax",
            peak_mode=0,
        )
        test_func = DataNormalization(settings=sets)
        test_func.forward(generate_test_data(do_tensor=False))


@pytest.mark.parametrize(
    "method, mode, expected_numpy, expected_torch",
    [
        (DataNormalizationTargets("zeroone"), 2, (0.0, 1.0), (0.0, 1.0)),
        (DataNormalizationTargets("minmax"), 2, (-1.0, 1.0), (-1.0, 1.0)),
        (
            DataNormalizationTargets("norm"),
            0,
            (-0.2537967223202087, 0.2539491764105914),
            (-0.2537967562675476, 0.25394919514656067),
        ),
        (
            DataNormalizationTargets("norm"),
            1,
            (-0.2537967223202087, 0.2539491764105914),
            (-0.2537967562675476, 0.25394919514656067),
        ),
        (
            DataNormalizationTargets("norm"),
            2,
            (-0.2537967223202087, 0.2539491764105914),
            (-0.2537967562675476, 0.25394919514656067),
        ),
        (
            DataNormalizationTargets("zscore"),
            0,
            (-1.4349982392766611, 1.4477661687200503),
            (-1.4349982738494873, 1.4477663040161133),
        ),
        (
            DataNormalizationTargets("zscore"),
            1,
            (-1.4349982392766611, 1.4477661687200503),
            (-1.4349982738494873, 1.4477663040161133),
        ),
        (
            DataNormalizationTargets("zscore"),
            2,
            (-1.4349982392766611, 1.4477661687200503),
            (-1.4349982738494873, 1.4477663040161133),
        ),
        (
            DataNormalizationTargets("medianmad"),
            0,
            (-1.451448498197262, 1.5791120924283681),
            (-1.4514487981796265, 1.5791120529174805),
        ),
        (
            DataNormalizationTargets("medianmad"),
            1,
            (-1.451448498197262, 1.5791120924283681),
            (-1.4514487981796265, 1.5791120529174805),
        ),
        (
            DataNormalizationTargets("medianmad"),
            2,
            (-1.451448498197262, 1.5791120924283681),
            (-1.4514487981796265, 1.5791120529174805),
        ),
        (
            DataNormalizationTargets("meanmad"),
            0,
            (-1.620773402838807, 1.620773402838807),
            (-1.620773434638977, 1.6207733154296875),
        ),
        (
            DataNormalizationTargets("meanmad"),
            1,
            (-1.620773402838807, 1.620773402838807),
            (-1.620773434638977, 1.6207733154296875),
        ),
        (
            DataNormalizationTargets("meanmad"),
            2,
            (-1.620773402838807, 1.620773402838807),
            (-1.620773434638977, 1.6207733154296875),
        ),
    ],
)
def test_normalization_method(
    method: DataNormalizationTargets,
    mode: int,
    expected_numpy: tuple[float],
    expected_torch: tuple[float],
):
    sets = DataNormalizationSettings(
        method=method,
        peak_mode=mode,
    )
    test_func = DataNormalization(settings=sets)

    din = generate_test_data(do_tensor=True)
    data0 = test_func.forward(din)
    result0 = (data0.min().float().item(), data0.max().float().item())
    data1 = test_func(DataTransient(data=din, sampling_rate=1.0)).data
    result1 = (data1.min().float().item(), data1.max().float().item())
    np.testing.assert_almost_equal(result0, expected_torch, decimal=6)
    np.testing.assert_almost_equal(result1, expected_torch, decimal=6)

    din = generate_test_data(do_tensor=False)
    data0 = test_func.forward(din)
    result0 = (data0.min(), data0.max())
    data1 = test_func(DataTransient(data=din, sampling_rate=1.0)).data
    result1 = (data1.min(), data1.max())
    np.testing.assert_almost_equal(result0, expected_numpy, decimal=6)
    np.testing.assert_almost_equal(result1, expected_numpy, decimal=6)


def test_create_c_minmax_absmax():
    sets = DataNormalizationSettings(
        method=DataNormalizationTargets("minmax"),
        peak_mode=2,
    )

    with TemporaryDirectory() as directory:
        path2save = Path(directory)
        DataNormalization(settings=sets).create_design(
            target="mcu", total_bitwidth=8, id="0", path2save=path2save, is_signed=False
        )

        assert {file.name for file in path2save.iterdir()} == {
            "normalization_minmax_0.h",
            "normalization_minmax_0.c",
            "normalization_minmax_template.h",
        }


def test_create_c_zscore():
    sets = DataNormalizationSettings(
        method=DataNormalizationTargets("zscore"),
        peak_mode=0,
    )

    with TemporaryDirectory() as directory:
        path2save = Path(directory)
        DataNormalization(settings=sets).create_design(
            target="mcu", total_bitwidth=8, id="0", path2save=path2save, is_signed=False
        )

        assert {file.name for file in path2save.iterdir()} == {
            "normalization_zscore_0.h",
            "normalization_zscore_0.c",
            "normalization_zscore_template.h",
        }


DATA = np.array([[1.0, -5.0, 3.0], [-2.0, 2.0, 0.5]])


@pytest.mark.parametrize(
    "peak_mode, expected",
    [
        (0, [3.0, 2.0]),
        (1, [5.0, 2.0]),
        (2, [5.0, 2.0]),
    ],
)
def test_numpy_peak(peak_mode, expected):
    sets = DataNormalizationSettings(
        method=DataNormalizationTargets("zscore"),
        peak_mode=peak_mode,
    )
    results = DataNormalization(sets)._get_data_peak_value_numpy(DATA)
    np.testing.assert_allclose(results, expected)


@pytest.mark.parametrize(
    "peak_mode, expected",
    [
        (0, [3.0, 2.0]),
        (1, [5.0, 2.0]),
        (2, [5.0, 2.0]),
    ],
)
def test_tensor_peak(peak_mode, expected):
    sets = DataNormalizationSettings(
        method=DataNormalizationTargets("zscore"),
        peak_mode=peak_mode,
    )
    result = DataNormalization(sets)._get_data_peak_value_tensor(torch.tensor(DATA))
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))
