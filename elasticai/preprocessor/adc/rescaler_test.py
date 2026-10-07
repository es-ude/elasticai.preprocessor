from copy import deepcopy

import numpy as np
import pytest
from elasticai.creator.arithmetic import int_arithmetic

from elasticai.preprocessor import BuildPlatformTargets, SequentialSignal, get_path_to_project

from .rescaler import Rescaler, RescalerSettings, RescalerSettingsDefault


@pytest.mark.parametrize("target", ["mem", "pcie"])
def test_create_design_wrong_target(target: str) -> None:
    sets: RescalerSettings = deepcopy(RescalerSettingsDefault)
    with pytest.raises(ValueError):
        Rescaler(settings=sets).create_design(id="0", target=target, path2save=get_path_to_project())


@pytest.mark.parametrize("target", list(BuildPlatformTargets))
def test_create_design_right_target(target: str) -> None:
    sets: RescalerSettings = deepcopy(RescalerSettingsDefault)

    if target not in [BuildPlatformTargets.Workstation.value, BuildPlatformTargets.MCU.value]:
        with pytest.raises(NotImplementedError):
            Rescaler(settings=sets).create_design(
                id="1", target=target.value, path2save=get_path_to_project()
            )


@pytest.mark.parametrize("data_type", ["posit", "float"])
def test_data_conversion_wrong_datatype_input(data_type: str) -> None:
    arith = int_arithmetic(total_bits=8, signed=True)
    sets: RescalerSettings = deepcopy(RescalerSettingsDefault)
    sets.method = data_type

    data = np.asarray(
        np.random.randint(low=arith.minimum_as_integer, high=arith.maximum_as_integer, size=(1001,))
    )

    with pytest.raises(ValueError):
        Rescaler(settings=sets).convert(x=data)


@pytest.mark.parametrize("method, data_type", [("fxp", "float")])
def test_data_conversion_wrong_input(method: str, data_type: str) -> None:
    arith = int_arithmetic(total_bits=8, signed=True)
    sets: RescalerSettings = deepcopy(RescalerSettingsDefault)
    sets.method = method

    data = np.asarray(
        np.random.randint(low=arith.minimum_as_integer, high=arith.maximum_as_integer, size=(1001,)),
        dtype=data_type,
    )

    with pytest.raises(AttributeError):
        Rescaler(settings=sets).convert(x=data)


@pytest.mark.parametrize("method", ["fxp"])
def test_data_conversion_right_input(method: str) -> None:
    arith = int_arithmetic(total_bits=8, signed=True)
    sets: RescalerSettings = deepcopy(RescalerSettingsDefault)
    sets.method = method

    data_in = np.asarray(
        np.random.randint(low=arith.minimum_as_integer, high=arith.maximum_as_integer, size=(1001,))
    )
    data_out0 = Rescaler(settings=sets).convert(x=data_in)
    data_out1 = Rescaler(settings=sets)(SequentialSignal(data=data_in, sample_rate=1000.0)).data
    np.testing.assert_array_equal(data_in, data_out0)
    np.testing.assert_array_equal(data_out0, data_out1)


"""
def test_create_stream_verilog(adc_sets: ResamplerSettings):
    sets: ResamplerSettings = deepcopy(adc_sets)
    with TemporaryDirectory() as directory:
        path2save = Path(directory)
        path2save.mkdir(parents=True, exist_ok=True)

        try:
            TransientResampler(sets).create_design_for_streaming(
                target="fpga", path2save=path2save, id="0"
            )
        except NotImplementedError:
            assert True
        else:
            assert False


def test_create_stream_c(adc_sets: ResamplerSettings):
    sets: ResamplerSettings = deepcopy(adc_sets)
    with TemporaryDirectory() as directory:
        path2save = Path(directory)
        path2save.mkdir(parents=True, exist_ok=True)

        TransientResampler(sets).create_design_for_streaming(target="mcu", path2save=path2save, id="1")

        files_check = ["adc_1.c", "adc_1.h", "adc_template.h"]
        files_check.sort()
        files_avai = [file.name for file in path2save.glob("*.*")]
        files_avai.sort()

        assert len(files_check) == len(files_avai)
        assert files_check == files_avai
"""
