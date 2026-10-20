from pathlib import Path

import numpy as np

from elasticai.preprocessor import PreprocessingSequential, get_path_to_project
from elasticai.preprocessor._data_plot import Data, load_smatable_data, plot_data
from elasticai.preprocessor.adc import (
    ConversionResampler,
    ConversionResamplerSettings,
    ConversionResamplerTargets,
)
from elasticai.preprocessor.sequential import CreateSequentialSettings


def process_data(data: Data) -> Data:
    sets = ConversionResamplerSettings(
        method=ConversionResamplerTargets.FromVoltage,
        total_bits=12,
        frac_bits=11,
        is_signed=True,
        is_int_output=False,
        srate_orig=data.sampling_rate,
        srate_new=2 * data.sampling_rate,
        vneg=-0.1,
        vpos=0.1,
    )

    dut = PreprocessingSequential(
        ConversionResampler(settings=sets),
    )
    out = dut(
        x=data.data,
        fs=data.sampling_rate,
    )
    dut.create_design(
        settings=CreateSequentialSettings(
            target="fpga",
            total_bitwidth=sets.total_bits,
            frac_bitwidth=sets.frac_bits,
            do_signed=sets.is_signed,
            path2build=get_path_to_project("build"),
        )
    )
    return Data(
        sampling_rate=out.sampling_rate,
        time=np.arange(start=0, stop=out.data.shape[-1], step=1) / out.sampling_rate,
        data=out.data,
        channel_id=data.channel_id,
    )


if __name__ == "__main__":
    data_orig = load_smatable_data(
        path2data=get_path_to_project()
        / "smatable-data"
        / "raw-data"
        / "sub-P001"
        / "ses-S001/knock/sub-P001_ses-S001_task-Default_run-001_knock.xdf"
    )
    data_proc = process_data(data_orig)
    plot_data(
        data_orig=data_orig,
        data_filt=data_proc,
        path2save=Path(__file__).parent.absolute() / "example_adc.svg",
        yscale=1e2,
        ylabel_out="ADC out [100x FxP(12, 11)]",
    )
